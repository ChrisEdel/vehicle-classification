//
//  importViewController.swift
//  SceneDepthPointCloud
//
//  Created by Christian.Edelmayer on 16.03.21.
//  Copyright © 2021 Apple. All rights reserved.
//

import UIKit
import SceneKit


class importViewController: UIViewController {
    
    @IBOutlet weak var colorSwitch: UISwitch!
    @IBOutlet weak var confidenceControl: UISegmentedControl!
    var confidence: Int = 0
    private var maxValue: Float = Float(Int.min)
    private var minValue: Float = Float(Int.max)
    var points = [point]()
    var sceneView: SCNView = SCNView()
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        
        let scene = SCNScene()
        let geometry = makePointCloud()
        let pointCloudNode = SCNNode()
        
        // stop scanning in preview
        
        pointCloudNode.geometry = geometry
        scene.rootNode.addChildNode(pointCloudNode)
        
        self.sceneView = self.view as! SCNView
        self.sceneView.scene = scene
        
        self.sceneView.backgroundColor = UIColor.black
        self.sceneView.allowsCameraControl = true
    }
    // TODO: fix everything
    func makePointCloud() -> SCNGeometry {
        var vertices : [SCNVector3]   = []
        var colors   : [SIMD4<Float>] = []
        var indices  : [UInt32]       = []
        
        
        //print(points)
        for point in points {                // only preview points which meet to confidence threshold
            let confidenceThreshold = 2
            
            if(Int(point.confidence) >= confidenceThreshold) {
                vertices.append(SCNVector3(x: point.x, y: point.y, z: point.z))
                
                if point.y < minValue {
                    minValue = point.y
                }
                if point.y > maxValue {
                    maxValue = point.y
                }
                let temp = point.y - minValue// ((element[1] - minValue) / (maxValue - minValue))*1
                
                
                let frequency:Float = 2.5
                var red:Float   = (sin(frequency*temp + 0) * 127 + 128) / Float(255)
                var green:Float = (sin(frequency*temp + 2) * 127 + 128) / Float(255)
                var blue:Float  = (sin(frequency*temp + 4) * 127 + 128) / Float(255)
                
                if colorSwitch.isOn {
                    red = point.red
                    green = point.green
                    blue = point.blue
                }
                
                colors.append(SIMD4<Float>(red, green, blue, 1))
                indices.append(UInt32(vertices.count - 1))
            }
        }
        
        let vertexSource = SCNGeometrySource(vertices: vertices)
        let colorsData = Data(bytes: &colors, count: colors.count * MemoryLayout<SIMD4<Float>>.stride)
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
    
    @IBAction func confidenceValueChanged(_ sender: Any) {
        if confidence != confidenceControl.selectedSegmentIndex {
            confidence = confidenceControl.selectedSegmentIndex
            sceneView.scene!.rootNode.enumerateChildNodes { (node, stop) in
            node.removeFromParentNode() }
            let geometry = makePointCloud()
            let pointCloudNode = SCNNode()
            pointCloudNode.geometry = geometry
            sceneView.scene?.rootNode.addChildNode(pointCloudNode)
        }
    }
    
    @IBAction func colorClicked(_ sender: Any) {
        sceneView.scene!.rootNode.enumerateChildNodes { (node, stop) in
        node.removeFromParentNode() }
        let geometry = makePointCloud()
        let pointCloudNode = SCNNode()
        pointCloudNode.geometry = geometry
        sceneView.scene?.rootNode.addChildNode(pointCloudNode)
    }
    @IBAction func goBack(_ sender: Any) {
        dismiss(animated: true, completion: nil)
    }
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
