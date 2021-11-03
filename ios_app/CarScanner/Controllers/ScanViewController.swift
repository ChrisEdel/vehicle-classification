import UIKit
import Metal
import MetalKit
import ARKit
import MobileCoreServices
import UniformTypeIdentifiers
import JGProgressHUD


/// Controlls the scanner.
/// Allows to start/stop/reset a scan, go into settings/preview, name a scan, take a photo, export a scan and import/preview a ply file.
final class ScanViewController: UIViewController, ARSessionDelegate, RendererDelegate {
    
    // progress bar
    let hud = JGProgressHUD()
    
    private let session = ARSession()
    
    private var scanner: Scanner!
    
    private var settingsVC: SettingsViewController = SettingsViewController()
    
    // initally we do not scan
    var scanning = false
    
    @IBOutlet weak var scanButton: UIButton!
    @IBOutlet weak var exportButton: UIButton!
    
    @IBOutlet weak var usageInfo: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // returns MTLDevice -> The MTLDevice protocol defines the interface to a GPU
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        session.delegate = self
        
        hud.textLabel.text = "Loading"
        
        // set the view to use the default device
        if let view = view as? MTKView {
            view.device = device
            
            view.backgroundColor = UIColor.clear
            // we need this to enable depth test
            view.depthStencilPixelFormat = .depth32Float
            view.contentScaleFactor = 1
            view.delegate = self
            
            // configure the scanner to draw to the view
            scanner = Scanner(session: session, metalDevice: device, renderDestination: view)
            scanner.drawRectResized(size: view.bounds.size)
            scanner.delegate = self
        }
        
        // set up settings view controller
        self.settingsVC = UIStoryboard(name: "Main", bundle: nil).instantiateViewController(withIdentifier: "SettingsShowVC") as! SettingsViewController
        self.settingsVC.scanner = self.scanner
        self.settingsVC.usageInfoInViewController = self.usageInfo
        self.settingsVC.modalPresentationStyle = .fullScreen
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create a world-tracking configuration, and
        // enable the scene depth frame-semantic.
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = .sceneDepth
        
        // Run the view's session
        session.run(configuration)
        
        // The screen shouldn't dim during AR experiences.
        UIApplication.shared.isIdleTimerDisabled = true
    }
    
    /// Starts or stops the scan depending on the state of the scan. If the scan is active, the scan is stopped and otherwise.
    @IBAction func scanButtonClicked(_ sender: Any) {
        // hide usage info as soon as the scan is initally started (until the app is closed or the reset button was clicked)
        usageInfo.isHidden = true
        
        if(!scanning) {
            // notify the scanner to add the newly scanned points to its data
            self.scanner.updatePoints = true
            scanButton.setTitle("End", for: .normal)
        } else {
            scanButton.setTitle("Start", for: .normal)
        }
        
        scanning = !scanning
    }
    
    /// Resets the scan and deletes all the gathered data.
    @IBAction func resetButtonClicked(_ sender: Any) {
        // resetting automatically stops the scan
        scanning = false
        
        scanner.reset()
        
        scanButton.setTitle("Start", for: .normal)
        
        // show usage info again in order to let the user know he can start a new scan
        usageInfo.isHidden = false
    }
    
    /// Clicking the photo button allows the user to take a photo.
    @IBAction func photoButtonClicked(_ sender: Any) {
        // notify the scanner to take a photo
        self.scanner.takePhoto = true
        
        // inform the user
        let alert = UIAlertController(title: "", message: "Photo successfully taken", preferredStyle: .alert)
        self.present(alert, animated: true, completion: nil)
        
        // dismiss the alert after one second
        let when = DispatchTime.now() + 1
        DispatchQueue.main.asyncAfter(deadline: when){
            alert.dismiss(animated: true, completion: nil)
        }
    }
    
    /// Clicking the settings button presents the settings screen.
    @IBAction func settingsButtonClicked(_ sender: Any) {
        // entering the settings automatically stops the scan
        scanning = false
        
        scanButton.setTitle("Start", for: .normal)
        
        show(settingsVC, sender: self)
    }
    
    /// Clicking the prewview button presents the peview screen showing the point cloud gathered from the scan.
    @IBAction func previewButtonClicked(_ sender: Any) {
        // entering the preview automatically stops the scan
        scanning = false
        
        scanButton.setTitle("Start", for: .normal)
        
        // use the main thread for UI updates
        DispatchQueue.main.async{
            self.hud.show(in: self.view)
        }
        
        // work intensive methods do not necessarily need to be executed by the main thread
        DispatchQueue.global().async {
            self.scanner.savePoints()
            
            let previewVC = UIStoryboard(name: "Main", bundle: nil).instantiateViewController(withIdentifier: "PreviewShowVC") as! PreviewViewController
            previewVC.scanner = self.scanner
            // only show points with a high confidence in order to reduce memory and compute time
            previewVC.points = self.scanner.pointsFiltered
            previewVC.modalPresentationStyle = .fullScreen
            
            // main thread dismisses loading info
            DispatchQueue.main.async{
                self.hud.dismiss()
                self.show(previewVC, sender: self)
            }
        }
    }
    
    /// Clicking the name button allows the user to give the exported scan a name.
    @IBAction func nameButtonClicked(_ sender: Any) {
        let alert = UIAlertController(title: "What are you scanning?", message: "Please enter a name", preferredStyle: .alert)
        
        alert.addTextField { (textField) in
            textField.text = ""
        }
        
        alert.addAction(UIAlertAction(title: "OK", style: .default, handler: { [weak alert] (_) in
            let textField = alert?.textFields![0]
            self.scanner.name = (textField?.text)!
        }))
        
        self.present(alert, animated: true, completion: nil)
    }
    
    /// Clicking the export button allows the user to export the corresponding .ply file and all the images in a zip-archive.
    @IBAction func exportButtonClicked(_ sender: Any) {
        // use the main thread for UI updates
        DispatchQueue.main.async{
            self.hud.show(in: self.view)
        }
        
        // work intensive method does not necessarily need to be executed by the main thread
        DispatchQueue.global().async {
            self.scanner.exportToFile()
        }
    }
    
    /// Clicking the import button allows the user to import .ply files and preview their corresponding point clouds.
    @IBAction func importButtonClicked(_ sender: Any) {
        // importing automatically stops the scan
        scanning = false
        
        scanButton.setTitle("Start", for: .normal)
        
        let documentPicker = UIDocumentPickerViewController(forOpeningContentTypes: [UTType.threeDContent], asCopy: true)
        documentPicker.delegate = self
        documentPicker.allowsMultipleSelection = false
        documentPicker.modalPresentationStyle = .fullScreen
        
        present(documentPicker, animated: true, completion: { [weak self] in
            // main thread shows loading info after selecting a file
            DispatchQueue.main.async {
                self?.hud.show(in: (self?.view)!)
            }
        })
    }
    
    /// Delegate method for saving and exporting the file(s).
    func didFinishSaving(path: URL) {
        DispatchQueue.main.async {
            var ac:UIActivityViewController
            
            if (self.scanner.photosTaken) {
                // export ply file and zipped images
                guard let imagePath = self.scanner.imagePath else {
                    return
                }
                
                ac = UIActivityViewController(activityItems: [path, imagePath], applicationActivities: nil)
            } else {
                // export ply file only
                ac = UIActivityViewController(activityItems: [path], applicationActivities: nil)
            }
            
            self.hud.dismiss()
            
            self.present(ac, animated: true, completion: nil)
        }
    }
    
    // Auto-hide the home indicator to maximize immersion in AR experiences.
    override var prefersHomeIndicatorAutoHidden: Bool {
        return true
    }
    
    // Hide the status bar to maximize immersion in AR experiences.
    override var prefersStatusBarHidden: Bool {
        return true
    }
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user.
        guard error is ARError else { return }
        let errorWithInfo = error as NSError
        let messages = [
            errorWithInfo.localizedDescription,
            errorWithInfo.localizedFailureReason,
            errorWithInfo.localizedRecoverySuggestion
        ]
        let errorMessage = messages.compactMap({ $0 }).joined(separator: "\n")
        DispatchQueue.main.async {
            // Present an alert informing about the error that has occurred.
            let alertController = UIAlertController(title: "The AR session failed.", message: errorMessage, preferredStyle: .alert)
            let restartAction = UIAlertAction(title: "Restart Session", style: .default) { _ in
                alertController.dismiss(animated: true, completion: nil)
                if let configuration = self.session.configuration {
                    self.session.run(configuration, options: .resetSceneReconstruction)
                }
            }
            alertController.addAction(restartAction)
            self.present(alertController, animated: true, completion: nil)
        }
    }
}

// MARK: - MTKViewDelegate

extension ScanViewController: MTKViewDelegate {
    // Called whenever view changes orientation or layout is changed
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        scanner.drawRectResized(size: size)
    }
    
    // Called whenever the view needs to render
    func draw(in view: MTKView) {
        scanner.draw(recording: scanning)
    }
}

// MARK: - RenderDestinationProvider

protocol RenderDestinationProvider {
    var currentRenderPassDescriptor: MTLRenderPassDescriptor? { get }
    var currentDrawable: CAMetalDrawable? { get }
    var colorPixelFormat: MTLPixelFormat { get set }
    var depthStencilPixelFormat: MTLPixelFormat { get set }
    var sampleCount: Int { get set }
}

extension MTKView: RenderDestinationProvider {
    
}

// MARK: - UIDocumentPickerDelegate

extension ScanViewController: UIDocumentPickerDelegate {
    
    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        DispatchQueue.main.async {
            self.hud.dismiss()
        }
    }
    
    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let selectedFileURL = urls.first else {
            return
        }
        
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let sandboxFileURL = dir.appendingPathComponent(selectedFileURL.lastPathComponent)
        
        if FileManager.default.fileExists(atPath: sandboxFileURL.path) {
            print("Already exists! Do nothing")
        } else {
            do {
                try FileManager.default.copyItem(at: selectedFileURL, to: sandboxFileURL)
                print("Copied file!")
            }
            catch {
                print("Error: \(error)")
            }
        }
        
        do {
            let data = try String(contentsOfFile: sandboxFileURL.path, encoding: .utf8)
            
            let points = Parser.parsePly(plyString: data.components(separatedBy: .newlines))
            
            let previewVC = UIStoryboard(name: "Main", bundle: nil).instantiateViewController(withIdentifier: "PreviewShowVC") as! PreviewViewController
            previewVC.scanner = scanner
            previewVC.points = points
            previewVC.modalPresentationStyle = .fullScreen
            
            DispatchQueue.main.async {
                self.hud.dismiss()
                self.show(previewVC, sender: self)
            }
            
            // delete the sandbox file
            try FileManager.default.removeItem(at: sandboxFileURL)
        } catch {
            print(error)
        }
    }
}
