import UIKit

/// Controlls the settings view.
/// Allows the user to modify the number of points and the grid size.
class SettingsViewController: UIViewController {
    private let isUIEnabled = true
    
    var usageInfoInViewController:UILabel = UILabel()
    
    private let MAX_NUMBER_OF_POINTS = 10_000_000
    
    private let DEFAULT_MAX_POINTS         = 10_000_000
    private let DEFAULT_GRID_SIZE          = 5_000
    private let DEFAULT_DISTANCE_THRESHOLD = 2
    
    private var currentMaxNumberOfPoints = 0
    private var currentGridSize          = 0
    
    private var currentDistanceThreshold = 0
    private var customDistanceThreshold  = 0
    
    var scanner: Scanner!
    
    @IBOutlet weak var maxNumberOfPointsTextField: UITextField!
    @IBOutlet weak var gridSizeTextField: UITextField!
    
    @IBOutlet weak var distanceThresholdLabel: UILabel!
    @IBOutlet weak var distanceThresholdSlider: UISlider!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.maxNumberOfPointsTextField.delegate = self
        self.gridSizeTextField.delegate = self
        
        self.currentMaxNumberOfPoints = DEFAULT_MAX_POINTS
        self.currentGridSize = DEFAULT_GRID_SIZE
        
        self.maxNumberOfPointsTextField.text = String(currentMaxNumberOfPoints)
        self.gridSizeTextField.text = String(currentGridSize)
        
        self.currentDistanceThreshold = self.DEFAULT_DISTANCE_THRESHOLD
        self.customDistanceThreshold = self.currentDistanceThreshold
        
        self.distanceThresholdLabel.text = "Distance threshold: \(self.currentDistanceThreshold) m"
        self.distanceThresholdSlider.setValue(Float(self.currentDistanceThreshold), animated: false)
    }
    
    /// Resets the number of points and the grid size to the default values. Addtionally resets the scan and deletes all the gathered data.
    /// The user is informed about this and has the option to cancel the reset.
    @IBAction func resetToDefaults(_ sender: Any) {
        let resetAlert = UIAlertController(title: "Warning", message: "Resetting to default values will completely reset the scan, meaning all the data will be lost.", preferredStyle: UIAlertController.Style.alert)
        
        resetAlert.addAction(UIAlertAction(title: "Ok", style: .default, handler: { [weak self] (action: UIAlertAction!) in
            guard let strongSelf = self else {
                return
            }
            
            strongSelf.currentMaxNumberOfPoints = strongSelf.DEFAULT_MAX_POINTS
            strongSelf.currentGridSize = strongSelf.DEFAULT_GRID_SIZE
            
            strongSelf.maxNumberOfPointsTextField.text = String(strongSelf.currentMaxNumberOfPoints)
            strongSelf.gridSizeTextField.text = String(strongSelf.currentGridSize)
            
            strongSelf.currentDistanceThreshold = strongSelf.DEFAULT_DISTANCE_THRESHOLD
            strongSelf.customDistanceThreshold = strongSelf.currentDistanceThreshold
            
            strongSelf.distanceThresholdSlider.setValue(Float(strongSelf.currentDistanceThreshold), animated: true)
            strongSelf.distanceThresholdLabel.text = "Distance threshold: \(String(strongSelf.currentDistanceThreshold)) m"
            
            strongSelf.scanner.updateMaxPoints(threshold: strongSelf.currentMaxNumberOfPoints)
            strongSelf.scanner.updateGridSize(size: strongSelf.currentGridSize)
            strongSelf.scanner.updateDistanceThreshold(threshold: strongSelf.currentDistanceThreshold)
            
            strongSelf.scanner.reset()
            
            // show the usage info again since the scan was completely resetted
            strongSelf.usageInfoInViewController.isHidden = false
        }))
        
        resetAlert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { (action: UIAlertAction!) in
            // do nothing
        }))
        
        present(resetAlert, animated: true, completion: nil)
    }
    
    /// Allows the user to change the number of points and the grid size. Addtionally resets the scan and deletes all the gathered data.
    /// The user is informed about this and has the option to cancel.
    private func changeTextfieldValues(_ textField: UITextField) -> Bool {
        guard let newMaxNumberOfPoints = Int(maxNumberOfPointsTextField.text!) else {
            maxNumberOfPointsTextField.text = String(currentMaxNumberOfPoints)
            return true
        }
        
        // number of points cannot be less than zero
        if newMaxNumberOfPoints < 0 {
            maxNumberOfPointsTextField.text = String(currentMaxNumberOfPoints)
            
            let alert = UIAlertController(title: "Invalid argument", message: "Maximal number of points cannot be less than zero.", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "Ok", style: .default, handler: nil))
            present(alert, animated: true)
        }
        
        // number of points cannot be less than the grid size
        if newMaxNumberOfPoints < currentGridSize {
            maxNumberOfPointsTextField.text = String(currentMaxNumberOfPoints)
            
            let alert = UIAlertController(title: "Invalid argument", message: "Maximal number of points cannot be less than the grid size.", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "Ok", style: .default, handler: nil))
            present(alert, animated: true)
        }
        
        // number of points need to be below a certain threshold in order to prevent memory issues
        if newMaxNumberOfPoints > MAX_NUMBER_OF_POINTS {
            maxNumberOfPointsTextField.text = String(currentMaxNumberOfPoints)
            
            let alert = UIAlertController(title: "Invalid argument", message: "Maximal number of points cannot be greater than \(MAX_NUMBER_OF_POINTS).", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "Ok", style: .default, handler: nil))
            present(alert, animated: true)
        }
        
        guard let newGridSize = Int(gridSizeTextField.text!) else {
            gridSizeTextField.text = String(currentGridSize)
            return true
        }
        
        // grid size cannot be less than zero
        if newGridSize < 0 {
            gridSizeTextField.text = String(currentGridSize)
            
            let alert = UIAlertController(title: "Invalid argument", message: "Grid size cannot be less than zero.", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "Ok", style: .default, handler: nil))
            present(alert, animated: true)
        }
        
        // grid size cannot be greater than the number of points
        if newGridSize > currentMaxNumberOfPoints {
            gridSizeTextField.text = String(currentGridSize)
            
            let alert = UIAlertController(title: "Invalid argument", message: "Grid size cannot be greater than the maximal number of points.", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "Ok", style: .default, handler: nil))
            present(alert, animated: true)
        }
        
        let resetAlert = UIAlertController(title: "Warning", message: "Changing the values will completely reset the scan, meaning all the data will be lost.", preferredStyle: UIAlertController.Style.alert)
        
        resetAlert.addAction(UIAlertAction(title: "Ok", style: .default, handler: { [weak self] (action: UIAlertAction!) in
            guard let strongSelf = self else {
                return
            }
            
            strongSelf.currentMaxNumberOfPoints = newMaxNumberOfPoints
            strongSelf.currentGridSize = newGridSize
        
            strongSelf.maxNumberOfPointsTextField.text = String(strongSelf.currentMaxNumberOfPoints)
            strongSelf.gridSizeTextField.text = String(strongSelf.currentGridSize)
            
            strongSelf.scanner.updateMaxPoints(threshold: strongSelf.currentMaxNumberOfPoints)
            strongSelf.scanner.updateGridSize(size: strongSelf.currentGridSize)
            
            strongSelf.scanner.reset()
            
            strongSelf.usageInfoInViewController.isHidden = false
        }))
        
        resetAlert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { [weak self] (action: UIAlertAction!) in
            guard let strongSelf = self else {
                return
            }
            
            strongSelf.maxNumberOfPointsTextField.text = String(strongSelf.currentMaxNumberOfPoints)
            strongSelf.gridSizeTextField.text = String(strongSelf.currentGridSize)
        }))
        present(resetAlert, animated: true, completion: nil)
        
        return true
    }
    
    /// Allows the user to change the distance threshold.
    @IBAction func changeDistanceThreshold(_ sender: UISlider) {
        distanceThresholdLabel.text = "Distance threshold: \(String(Int(sender.value))) m"
        customDistanceThreshold = Int(sender.value)
    }
    
    /// Dismisses the current view, i.e. taking the user back to the view before.
    /// If the distance threshold has been updated, going back addtionally resets the scan and deletes all the gathered data.
    /// The user is informed about this and has the option to cancel.
    @IBAction func goBack(_ sender: Any) {
        if (customDistanceThreshold != currentDistanceThreshold) {
            let resetAlert = UIAlertController(title: "Warning", message: "Changing the distance threshold values will completely reset the scan, meaning all the data will be lost.", preferredStyle: UIAlertController.Style.alert)
            
            resetAlert.addAction(UIAlertAction(title: "Ok", style: .default, handler: { [weak self] (action: UIAlertAction!) in
                guard let strongSelf = self else {
                    return
                }
                
                strongSelf.currentDistanceThreshold = strongSelf.customDistanceThreshold
                
                strongSelf.scanner.updateDistanceThreshold(threshold: strongSelf.currentDistanceThreshold)
                
                strongSelf.scanner.reset()
                
                // show the usage info again since the scan was completely resetted
                strongSelf.usageInfoInViewController.isHidden = false
                
                strongSelf.dismiss(animated: true, completion: nil)
            }))
            
            resetAlert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { [weak self] (action: UIAlertAction!) in
                guard let strongSelf = self else {
                    return
                }
                
                strongSelf.customDistanceThreshold = strongSelf.currentDistanceThreshold
                
                strongSelf.distanceThresholdLabel.text = "Distance threshold: \(String(strongSelf.currentDistanceThreshold)) m"
                strongSelf.distanceThresholdSlider.setValue(Float(strongSelf.currentDistanceThreshold), animated: true)
                
                strongSelf.dismiss(animated: true, completion: nil)
            }))
            
            present(resetAlert, animated: true, completion: nil)
        } else {
            dismiss(animated: true, completion: nil)
        }
    }
}

// MARK: - UITextFieldDelegate

extension SettingsViewController: UITextFieldDelegate {
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder()
        
        return changeTextfieldValues(textField)
    }
}
