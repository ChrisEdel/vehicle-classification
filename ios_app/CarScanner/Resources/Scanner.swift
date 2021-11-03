import Metal
import MetalKit
import ARKit
import JGProgressHUD

/// Scanner using the LiDAR sensor in order to scan objects (mainly cars).
final class Scanner {
    // maximum number of points we store in the point cloud
    private var maxPoints = 10_000_000
    // number of sample points on the grid
    private var numGridPoints = 5_000
    // particle's size in pixels
    private let particleSize: Float = 10
    // we only use portrait orientation in this app
    private let orientation = UIInterfaceOrientation.portrait
    // camera's threshold values for detecting when the camera moves so that we can accumulate the points
    private let cameraRotationThreshold = cos(2 * .degreesToRadian)
    private let cameraTranslationThreshold: Float = pow(0.02, 2)   // (meter-squared)
    // the max number of command buffers in flight
    private let maxInFlightBuffers = 3
    // distance threshold
    private var distanceThreshold = 2
    
    private lazy var rotateToARCamera = Self.makeRotateToARCameraMatrix(orientation: orientation)
    private let session: ARSession
    weak var delegate: RendererDelegate?
    
    var globalViewMatrix: simd_float4x4 = simd_float4x4()
    var isGlobalViewMatrixSet = false
    
    // metal objects and textures
    private let device: MTLDevice
    private let library: MTLLibrary
    private let renderDestination: RenderDestinationProvider
    private let relaxedStencilState: MTLDepthStencilState
    private let depthStencilState: MTLDepthStencilState
    private let commandQueue: MTLCommandQueue
    private lazy var unprojectPipelineState = makeUnprojectionPipelineState()!
    private lazy var rgbPipelineState = makeRGBPipelineState()!
    private lazy var particlePipelineState = makeParticlePipelineState()!
    // texture cache for captured image
    private lazy var textureCache = makeTextureCache()
    private var capturedImageTextureY: CVMetalTexture?
    private var capturedImageTextureCbCr: CVMetalTexture?
    private var depthTexture: CVMetalTexture?
    private var confidenceTexture: CVMetalTexture?
    
    // multi-buffer rendering pipeline
    private let inFlightSemaphore: DispatchSemaphore
    private var currentBufferIndex = 0
    
    // the current viewport size
    private var viewportSize = CGSize()
    // the grid of sample points
    private lazy var gridPointsBuffer = MetalBuffer<Float2>(device: device,
                                                            array: makeGridPoints(),
                                                            index: kGridPoints.rawValue, options: [])
    
    // RGB buffer
    private lazy var rgbUniforms: RGBUniforms = {
        var uniforms = RGBUniforms()
        uniforms.viewToCamera.copy(from: viewToCamera)
        uniforms.viewRatio = Float(viewportSize.width / viewportSize.height)
        return uniforms
    }()
    // point cloud buffer
    private lazy var pointCloudUniforms: PointCloudUniforms = {
        var uniforms = PointCloudUniforms()
        uniforms.maxPoints = Int32(maxPoints)
        uniforms.confidenceThreshold = Int32(confidenceThreshold)
        uniforms.distanceThreshold = Int32(distanceThreshold)
        uniforms.particleSize = particleSize
        uniforms.cameraResolution = cameraResolution
        return uniforms
    }()
    private var pointCloudUniformsBuffers = [MetalBuffer<PointCloudUniforms>]()
    // particles buffer
    var particlesBuffer: MetalBuffer<ParticleUniforms>
    private var currentPointIndex = 0
    var currentPointCount = 0
    
    // camera data
    private var sampleFrame: ARFrame { session.currentFrame! }
    private lazy var cameraResolution = Float2(Float(sampleFrame.camera.imageResolution.width), Float(sampleFrame.camera.imageResolution.height))
    private lazy var viewToCamera = sampleFrame.displayTransform(for: orientation, viewportSize: viewportSize).inverted()
    private lazy var lastCameraTransform = sampleFrame.camera.transform
    
    // interfaces
    var confidenceThreshold = 2 {
        didSet {
            // apply the change for the shader
            pointCloudUniforms.confidenceThreshold = Int32(confidenceThreshold)
        }
    }
    
    var recording = false
    
    var isSavingFile = false
    
    var photoHelperCounter = 0
    
    var name: String = "scan"
    
    var points = [point]()
    var pointsFiltered = [point]()
    var updatePoints = true
    
    var photosTaken = false
    var takePhoto = false
    
    var currentImage = UIImage()
    var images = [Data]()
    var imagePath: URL?
    
    init(session: ARSession, metalDevice device: MTLDevice, renderDestination: RenderDestinationProvider) {
        self.session = session
        self.device = device
        self.renderDestination = renderDestination
        
        library = device.makeDefaultLibrary()!
        commandQueue = device.makeCommandQueue()!
        
        // initialize our buffers
        for _ in 0 ..< maxInFlightBuffers {
            pointCloudUniformsBuffers.append(.init(device: device, count: 1, index: kPointCloudUniforms.rawValue))
        }
        particlesBuffer = .init(device: device, count: maxPoints, index: kParticleUniforms.rawValue)
        
        // rbg does not need to read/write depth
        let relaxedStateDescriptor = MTLDepthStencilDescriptor()
        relaxedStencilState = device.makeDepthStencilState(descriptor: relaxedStateDescriptor)!
        
        // setup depth test for point cloud
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = .lessEqual
        depthStateDescriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthStateDescriptor)!
        
        inFlightSemaphore = DispatchSemaphore(value: maxInFlightBuffers)
    }
    
    func drawRectResized(size: CGSize) {
        viewportSize = size
    }
    
    /// Updates the textures of the captured image.
    private func updateCapturedImageTextures(frame: ARFrame) {
        // create two textures (Y and CbCr) from the provided frame's captured image
        let pixelBuffer = frame.capturedImage
        guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 2 else {
            return
        }
        
        capturedImageTextureY = makeTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .r8Unorm, planeIndex: 0)
        capturedImageTextureCbCr = makeTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .rg8Unorm, planeIndex: 1)
    }
    
    /// Updates the depth textures.
    private func updateDepthTextures(frame: ARFrame) -> Bool {
        guard let depthMap = frame.sceneDepth?.depthMap,
              let confidenceMap = frame.sceneDepth?.confidenceMap else {
            return false
        }
        
        depthTexture = makeTexture(fromPixelBuffer: depthMap, pixelFormat: .r32Float, planeIndex: 0)
        confidenceTexture = makeTexture(fromPixelBuffer: confidenceMap, pixelFormat: .r8Uint, planeIndex: 0)
        return true
    }
    
    /// Updates the frame dependent info and the pointCloudUniforms buffer.
    private func update(frame: ARFrame) {
        // frame dependent info
        let camera = frame.camera
        let cameraIntrinsicsInversed = camera.intrinsics.inverse
        let viewMatrix = camera.viewMatrix(for: orientation)
        let viewMatrixInversed = viewMatrix.inverse
        let projectionMatrix = camera.projectionMatrix(for: orientation, viewportSize: viewportSize, zNear: 0.001, zFar: 0)
        pointCloudUniforms.viewProjectionMatrix = projectionMatrix * viewMatrix
        pointCloudUniforms.localToWorld = viewMatrixInversed * rotateToARCamera
        pointCloudUniforms.cameraIntrinsicsInversed = cameraIntrinsicsInversed
    }
    
    /// Draws and renders the particles.
    ///
    /// - Parameter recording: If true, the newly accumulated particles are rendered. Otherwise not.
    func draw(recording: Bool) {
        self.recording = recording
        
        guard let currentFrame = session.currentFrame,
              let renderDescriptor = renderDestination.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderDescriptor) else {
            return
        }
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        commandBuffer.addCompletedHandler { [weak self] commandBuffer in
            if let self = self {
                self.inFlightSemaphore.signal()
            }
        }
        
        // update frame data
        update(frame: currentFrame)
        updateCapturedImageTextures(frame: currentFrame)
        
        // handle buffer rotating
        currentBufferIndex = (currentBufferIndex + 1) % maxInFlightBuffers
        pointCloudUniformsBuffers[currentBufferIndex][0] = pointCloudUniforms
        
        // the user wants to explicitly take a photo
        if (takePhoto) {
            takePhoto = !takePhoto
            self.currentImage = UIImage(ciImage: CIImage(cvPixelBuffer: currentFrame.capturedImage))
            currentImage = currentImage.imageRotated(by: CGFloat(Double.pi/2))
            images.append(self.currentImage.jpegData(compressionQuality: 0.85)!)
        }
        
        if (recording) {
            if shouldAccumulate(frame: currentFrame), updateDepthTextures(frame: currentFrame) {
                accumulatePoints(frame: currentFrame, commandBuffer: commandBuffer, renderEncoder: renderEncoder)
            } else {
                photoHelperCounter += 1
                // take photos automatically under the assumption that there are no quick camera movements
                if photoHelperCounter == 100 {
                    self.currentImage = UIImage(ciImage: CIImage(cvPixelBuffer: currentFrame.capturedImage))
                    currentImage = currentImage.imageRotated(by: CGFloat(Double.pi/2))
                    images.append(self.currentImage.jpegData(compressionQuality: 0.85)!)
                    
                    photoHelperCounter = 0
                }
            }
        }
        
        // render particles
        renderEncoder.setDepthStencilState(depthStencilState)
        renderEncoder.setRenderPipelineState(particlePipelineState)
        renderEncoder.setVertexBuffer(pointCloudUniformsBuffers[currentBufferIndex])
        renderEncoder.setVertexBuffer(particlesBuffer)
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: currentPointCount)
        renderEncoder.endEncoding()
        
        commandBuffer.present(renderDestination.currentDrawable!)
        commandBuffer.commit()
    }
    
    /// Checks whether we should accumulate based on camera movement.
    ///
    /// - Parameter frame: The current frame.
    /// - Returns: True, if we should accumulate. False otherwise.
    private func shouldAccumulate(frame: ARFrame) -> Bool {
        let cameraTransform = frame.camera.transform
        return currentPointCount == 0
            || dot(cameraTransform.columns.2, lastCameraTransform.columns.2) <= cameraRotationThreshold
            || distance_squared(cameraTransform.columns.3, lastCameraTransform.columns.3) >= cameraTranslationThreshold
    }
    
    /// Accumulates the points.
    private func accumulatePoints(frame: ARFrame, commandBuffer: MTLCommandBuffer, renderEncoder: MTLRenderCommandEncoder) {
        pointCloudUniforms.pointCloudCurrentIndex = Int32(currentPointIndex)
        
        var retainingTextures = [capturedImageTextureY, capturedImageTextureCbCr, depthTexture, confidenceTexture]
        commandBuffer.addCompletedHandler { buffer in
            retainingTextures.removeAll()
        }
        
        renderEncoder.setDepthStencilState(relaxedStencilState)
        renderEncoder.setRenderPipelineState(unprojectPipelineState)
        renderEncoder.setVertexBuffer(pointCloudUniformsBuffers[currentBufferIndex])
        renderEncoder.setVertexBuffer(particlesBuffer)
        renderEncoder.setVertexBuffer(gridPointsBuffer)
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(capturedImageTextureY!), index: Int(kTextureY.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(capturedImageTextureCbCr!), index: Int(kTextureCbCr.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(depthTexture!), index: Int(kTextureDepth.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(confidenceTexture!), index: Int(kTextureConfidence.rawValue))
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: gridPointsBuffer.count)
        
        currentPointIndex = (currentPointIndex + gridPointsBuffer.count) % maxPoints
        currentPointCount = min(currentPointCount + gridPointsBuffer.count, maxPoints)
        lastCameraTransform = frame.camera.transform
    }
    
// MARK: - Useful settings methods
    
    /// Updates the grid size.
    ///
    /// - Parameter size: The new grid size.
    func updateGridSize(size: Int) {
        numGridPoints = size
        
        gridPointsBuffer = MetalBuffer<Float2>(device: device,
                                               array: makeGridPoints(),
                                               index: kGridPoints.rawValue, options: [])
    }
    
    /// Updates the upper limit of max points.
    ///
    /// - Parameter threshold: The new limit.
    func updateMaxPoints(threshold: Int) {
        maxPoints = threshold
        pointCloudUniforms.maxPoints = Int32(maxPoints)
    }
    
    func updateDistanceThreshold(threshold: Int) {
        distanceThreshold = threshold
        pointCloudUniforms.distanceThreshold = Int32(threshold)
    }
    
    /// Resets the whole scanner.
    func reset() {
        updatePoints = true
        
        points = [point]()
        pointsFiltered = [point]()
        
        currentBufferIndex = 0
        currentPointIndex = 0
        currentPointCount = 0
        
        // initialize our buffers
        for i in 0 ..< maxInFlightBuffers {
            pointCloudUniformsBuffers[i] = .init(device: device, count: 1, index: kPointCloudUniforms.rawValue)
        }
        particlesBuffer = .init(device: device, count: maxPoints, index: kParticleUniforms.rawValue)
        
        photosTaken = false
        
        images = [Data]()
        name = "scan"
    }
    
    /// Converts the scanned data into points and stores these points into an array in order for the points to be easily accessible.
    func savePoints() {
        // check if new points have been scanned, otherwise do nothing
        if(updatePoints) {
            updatePoints = false
            
            // we need to iterate over all the points because it is possible that we exceeded out threshold and therefore discarded older points
            for i in 0..<currentPointCount {
                let currentPoint = particlesBuffer[i]
                
                // store filtered points in extra array
                if (currentPoint.confidence >= 2) {
                    pointsFiltered.append(point(x: currentPoint.position.x, y: currentPoint.position.y, z: currentPoint.position.z, red: currentPoint.color.x, green: currentPoint.color.y, blue: currentPoint.color.z, confidence: currentPoint.confidence))
                }
                
                // ignore points with only zero values
                if (currentPoint.position.x == 0 && currentPoint.position.y == 0 && currentPoint.position.z == 0 && currentPoint.color.x == 0 && currentPoint.color.y == 0 && currentPoint.color.z == 0 && currentPoint.confidence == 0) {
                    continue
                }
                
                points.append(point(x: currentPoint.position.x, y: currentPoint.position.y, z: currentPoint.position.z, red: currentPoint.color.x, green: currentPoint.color.y, blue: currentPoint.color.z, confidence: currentPoint.confidence))
            }
        }
    }
    
    /// Exports the gathered data (point cloud and images).
    func exportToFile() {
        guard !self.isSavingFile else {
            return
        }
        
        self.isSavingFile = true
        
        let fileName = "\(name)_\(getDate())"
        
        var fileToWrite = ""
        
        let headers = ["ply", "format ascii 1.0", "element vertex \(currentPointCount)", "max number of points: \(maxPoints)", "grid size: \(numGridPoints)", "distance threshold: \(distanceThreshold)", "property float x", "property float y", "property float z", "property float red", "property float green", "property float blue", "property float quality", "element face 0", "property list uchar int vertex_indices", "end_header"]
        
        for header in headers {
            fileToWrite += header
            fileToWrite += "\n"
        }
        
        savePoints()
        
        for currentPoint in points {
            let pvValue = "\(currentPoint.x) \(currentPoint.y) \(currentPoint.z) \(currentPoint.red) \(currentPoint.green) \(currentPoint.blue) \(currentPoint.confidence)"
            fileToWrite += pvValue
            fileToWrite += "\n"
        }
        
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let documentsDirectory = paths[0]
        let file = documentsDirectory.appendingPathComponent("\(fileName).ply")
        
        if images.count > 0 {
            photosTaken = true
            zipImages(data: images, fileName: fileName, completion: { [weak self] url in
                self?.imagePath = url
            })
        }
        
        do {
            try fileToWrite.write(to: file, atomically: true, encoding: String.Encoding.ascii)
            isSavingFile = false
            delegate?.didFinishSaving(path: file)
            
            // remove file
            //try FileManager.default.removeItem(at: file)
        } catch {
            print("Failed to write PLY file", error)
        }
    }
}

// MARK: - Metal Helpers

private extension Scanner {
    func makeUnprojectionPipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "unprojectVertex") else {
            return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.isRasterizationEnabled = false
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    func makeRGBPipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "rgbVertex"),
              let fragmentFunction = library.makeFunction(name: "rgbFragment") else {
            return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    func makeParticlePipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "particleVertex"),
              let fragmentFunction = library.makeFunction(name: "particleFragment") else {
            return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    /// Makes sample points on camera image, also precompute the anchor point for animation
    func makeGridPoints() -> [Float2] {
        let gridArea = cameraResolution.x * cameraResolution.y
        let spacing = sqrt(gridArea / Float(numGridPoints))
        let deltaX = Int(round(cameraResolution.x / spacing))
        let deltaY = Int(round(cameraResolution.y / spacing))
        
        var points = [Float2]()
        for gridY in 0 ..< deltaY {
            let alternatingOffsetX = Float(gridY % 2) * spacing / 2
            for gridX in 0 ..< deltaX {
                let cameraPoint = Float2(alternatingOffsetX + (Float(gridX) + 0.5) * spacing, (Float(gridY) + 0.5) * spacing)
                points.append(cameraPoint)
            }
        }
        
        return points
    }
    
    func makeTextureCache() -> CVMetalTextureCache {
        // Create captured image texture cache
        var cache: CVMetalTextureCache!
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        
        return cache
    }
    
    func makeTexture(fromPixelBuffer pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat, planeIndex: Int) -> CVMetalTexture? {
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)
        
        var texture: CVMetalTexture? = nil
        let status = CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, pixelBuffer, nil, pixelFormat, width, height, planeIndex, &texture)
        
        if status != kCVReturnSuccess {
            texture = nil
        }
        
        return texture
    }
    
    static func cameraToDisplayRotation(orientation: UIInterfaceOrientation) -> Int {
        switch orientation {
        case .landscapeLeft:
            return 180
        case .portrait:
            return 90
        case .portraitUpsideDown:
            return -90
        default:
            return 0
        }
    }
    
    static func makeRotateToARCameraMatrix(orientation: UIInterfaceOrientation) -> matrix_float4x4 {
        // flip to ARKit Camera's coordinate
        let flipYZ = matrix_float4x4(
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1] )
        
        let rotationAngle = Float(cameraToDisplayRotation(orientation: orientation)) * .degreesToRadian
        return flipYZ * matrix_float4x4(simd_quaternion(rotationAngle, Float3(0, 0, 1)))
    }
}

protocol RendererDelegate: class {
    func didFinishSaving(path: URL)
}
