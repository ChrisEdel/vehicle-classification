/*
 See LICENSE folder for this sample’s licensing information.
 
 Abstract:
 General Helper methods and properties
 */

import ARKit
import Zip

typealias Float2 = SIMD2<Float>
typealias Float3 = SIMD3<Float>
typealias Float4 = SIMD4<Float>

extension Float {
    static let degreesToRadian = Float.pi / 180
}

extension matrix_float3x3 {
    mutating func copy(from affine: CGAffineTransform) {
        columns.0 = Float3(Float(affine.a), Float(affine.c), Float(affine.tx))
        columns.1 = Float3(Float(affine.b), Float(affine.d), Float(affine.ty))
        columns.2 = Float3(0, 0, 1)
    }
}

func translationMatrix(x: Float, y: Float, z: Float) -> simd_float4x4 {
    let row1 = Float4(1, 0, 0, x)
    let row2 = Float4(0, 1, 0, y)
    let row3 = Float4(0, 0, 1, z)
    let row4 = Float4(0, 0, 0, 1)
    return .init(rows: [row1, row2, row3, row4])
}

/// Creates a temporary directory from a given filename.
///
/// - Parameter fileName: Given filename.
/// - Returns: URL of the temporary directory
func createTempDirectory(fileName: String) -> URL? {
    if let documentDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
        let dir = documentDirectory.appendingPathComponent("\(fileName)")
        do {
            try FileManager.default.createDirectory(atPath: dir.path, withIntermediateDirectories: true, attributes: nil)
        } catch {
            print(error.localizedDescription)
        }
        return dir
    } else {
        return nil
    }
}

/// Saves images in a directory.
///
/// - Parameter data: Images as `Data`
/// - Parameter fileName: Given filename.
/// - Returns: URL of the directory
func saveImages(data: [Data], fileName: String) -> URL? {
    guard let directory = createTempDirectory(fileName: fileName) else {
        return nil
    }
    
    do {
        for (i, imageData) in data.enumerated() {
            try imageData.write(to: directory.appendingPathComponent("\(i).jpg"))
        }
        return directory
    } catch {
        return nil
    }
}

/// Returns the current Date in the format  `"yyyy-MM-dd-HH-mm-ss"`.
func getDate() -> String {
    let date = Date()
    let df = DateFormatter()
    df.dateFormat = "yyyy-MM-dd-HH-mm-ss"
    return df.string(from: date)
}

/// Zipping given images together.
///
/// - Parameter data: Images as `Data`.
/// - Parameter fileName: Given filename.
func zipImages(data: [Data], fileName: String, completion: @escaping ((URL?) -> ())) {
    DispatchQueue.main.async {
        guard let directory = saveImages(data: data, fileName: fileName) else {
            completion(nil)
            return
        }
        do {
            let zipFilePath = try Zip.quickZipFiles([directory], fileName: "\(fileName)")
            completion(zipFilePath)
        } catch {
            completion(nil)
        }
    }
}

extension UIImage{
    /// Rotates image by given radian.
    ///
    /// - Parameter radian: Given radian.
    /// - Returns: Image rotated by `radian`.
    func imageRotated(by radian: CGFloat) -> UIImage{
        let rotatedSize = CGRect(origin: .zero, size: size)
            .applying(CGAffineTransform(rotationAngle: radian))
            .integral.size
        UIGraphicsBeginImageContext(rotatedSize)
        if let context = UIGraphicsGetCurrentContext() {
            let origin = CGPoint(x: rotatedSize.width / 2.0,
                                 y: rotatedSize.height / 2.0)
            context.translateBy(x: origin.x, y: origin.y)
            context.rotate(by: radian)
            draw(in: CGRect(x: -origin.y, y: -origin.x,
                            width: size.width, height: size.height))
            let rotatedImage = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()
            
            return rotatedImage ?? self
        }
        
        return self
    }
}
