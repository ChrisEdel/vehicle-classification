import Foundation

/// Parser for parsing files/strings.
final class Parser {
    
    /// Method for parsing the ply format given as an array of Strings.
    ///
    /// - Parameter plyString: Given array of Strings
    /// - Returns: Array of points
    static func parsePly(plyString: [String]) -> [point] {
        var points = [point] ()
        
        var inHeader = true
        
        for (_, line) in plyString.enumerated() {
            // parsing starts after header ends
            if (line == "end_header") {
                inHeader = false
                continue
            }
            
            // ignore the header
            if (inHeader) {
                continue
            }
            
            // use autoreleasepool in order to prevent memory issues
            autoreleasepool {
                let pointValues = line.components(separatedBy: " ")
                
                // ignore lines/points with missing values
                if (pointValues.count >= 6) {
                    // do not append points with low/medium confidence due to memory reasons
                    if (Float(pointValues[6])! >= 2) {
                        guard let x = Float(pointValues[0]),
                              let y = Float(pointValues[1]),
                              let z = Float(pointValues[2]),
                              let red = Float(pointValues[3]),
                              let green = Float(pointValues[4]),
                              let blue = Float(pointValues[5]),
                              let confidence = Float(pointValues[6]) else {
                            print("Error while parsing points")
                            return
                        }
                        
                        let currentPoint = point(x: x, y: y, z: z, red: red, green: green, blue: blue, confidence: confidence)
                        
                        points.append(currentPoint)
                    }
                }
            }
        }
        
        return points
    }
}
