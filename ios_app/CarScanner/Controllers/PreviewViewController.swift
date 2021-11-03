import UIKit
import SceneKit

/// Controlls the preview.
/// Allows to interactively preview a point cloud with the possibility of switching between the original color and a sampled color.
class PreviewViewController: UIViewController {
    
    @IBOutlet weak var colorSwitch: UISwitch!
    
    var scanner: Scanner!
    
    var sceneView: SCNView = SCNView()
    
    var confidence: Int = 0
    
    private var maxValue: Float = Float(Int.min)
    private var minValue: Float = Float(Int.max)
    
    private var colorReal = true
    
    var points: [point] = []
    
    var vertices: [SCNVector3] = []
    
    var indices: [UInt32] = []
    
    var colorsReal   : [SIMD4<Float>] = []
    var colorsSample : [SIMD4<Float>] = []
    var currentColors: [SIMD4<Float>] = []
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // stop scanning in preview
        self.scanner.recording = false
        
        let scene = SCNScene()
        
        // creating the vertices and colors arrays from the given points
        prepareData()
        
        // setting up the scene
        let geometry = buildPointCloudGeometry(vertices: vertices, colors: colorsReal, indices: indices)
        let pointCloudNode = SCNNode()
        
        pointCloudNode.geometry = geometry
        scene.rootNode.addChildNode(pointCloudNode)
        
        self.sceneView = self.view as! SCNView
        self.sceneView.scene = scene
        
        self.sceneView.backgroundColor = UIColor.darkGray
        self.sceneView.allowsCameraControl = true
    }
    
    /// Prepares all the necessary data for the creation of the scene. Creates the vertices, colors (both real and sample) and indices arrays.
    private func prepareData() {
        for point in points {
            vertices.append(SCNVector3(x: point.x, y: point.y, z: point.z))
            colorsReal.append(SIMD4<Float>(point.red, point.green, point.blue, 1))
            
            if point.y < minValue {
                minValue = point.y
            }
            
            if point.y > maxValue {
                maxValue = point.y
            }
            
            let temp = point.y - minValue
            
            let frequency: Float = 2.5
            let red:       Float = (sin(frequency*temp + 0) * 127 + 128) / Float(255)
            let green:     Float = (sin(frequency*temp + 2) * 127 + 128) / Float(255)
            let blue:      Float = (sin(frequency*temp + 4) * 127 + 128) / Float(255)
            
            colorsSample.append(SIMD4<Float>(red, green, blue, 1))
            
            indices.append(UInt32(vertices.count - 1))
        }
        
        currentColors = colorsReal
    }
    
    /// Creates and builds the geometry from given vertices, colors and indices.
    ///
    /// - Parameter vertices: Corresponding vertices.
    /// - Parameter colors:   Corresponding colors.
    /// - Parameter vertices: Corresponding indices.
    /// - Returns: A`SCNGeometry` containing the point cloud.
    private func buildPointCloudGeometry(vertices: [SCNVector3], colors: [SIMD4<Float>], indices: [UInt32]) -> SCNGeometry {
        let vertexSource = SCNGeometrySource(vertices: vertices)
        let colorsData = Data(bytes: colors, count: colors.count * MemoryLayout<SIMD4<Float>>.stride)
        let colorSource = SCNGeometrySource(data: colorsData, semantic: .color, vectorCount: colors.count, usesFloatComponents: true, componentsPerVector: 4, bytesPerComponent: MemoryLayout<Float>.stride, dataOffset: 0, dataStride: MemoryLayout<SIMD4<Float>>.stride)
        
        let pointCloudElement = SCNGeometryElement(indices: indices,
                                                   primitiveType: .point)
        pointCloudElement.pointSize = 0.01
        pointCloudElement.minimumPointScreenSpaceRadius = 2
        pointCloudElement.maximumPointScreenSpaceRadius = 10
        
        
        let geometry = SCNGeometry(sources: [vertexSource, colorSource],
                                   elements: [pointCloudElement])
        
        let material = SCNMaterial()
        material.diffuse.contents = #colorLiteral(red: 1, green: 1, blue: 1, alpha: 1)
        material.lightingModel = SCNMaterial.LightingModel.constant
        geometry.firstMaterial = material
        
        return geometry
    }
    
    /// Allows the user to change the color of the vertices (switch between real colors and sample colors).
    @IBAction func colorClicked(_ sender: Any) {
        colorReal = !colorReal
        
        sceneView.scene!.rootNode.enumerateChildNodes { (node, stop) in
            node.removeFromParentNode()
        }
        
        var geometry:SCNGeometry
        
        if (colorReal) {
            currentColors = colorsReal
        } else {
            currentColors = colorsSample
        }
        
        geometry = buildPointCloudGeometry(vertices: vertices, colors: currentColors, indices: indices)
        
        let pointCloudNode = SCNNode()
        
        pointCloudNode.geometry = geometry
        sceneView.scene?.rootNode.addChildNode(pointCloudNode)
    }
    
    /// Dismisses the current view, i.e. taking the user back to the view before.
    @IBAction func goBack(_ sender: Any) {
        dismiss(animated: true, completion: nil)
    }
}
