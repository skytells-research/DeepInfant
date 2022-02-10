//
//  ResultsObserver.swift
//  DeepInfant
//
//  Created by Hazem Ali on 2/10/22.
//
import UIKit
import Foundation
import SoundAnalysis

// Observer object that is called as analysis results are found.
@available(iOS 13.0, *)
class ResultsObserver : NSObject, SNResultsObserving {
    
    weak var vc: ViewController?
    
    func request(_ request: SNRequest, didProduce result: SNResult) {
        
        // Get the top classification.
        guard let result = result as? SNClassificationResult,
            let classification = result.classifications.first else { return }
        
        // Determine the time of this result.
        let formattedTime = String(format: "%.2f", result.timeRange.start.seconds)
        print("Analysis result for audio at time: \(formattedTime)")
        
        let confidence = classification.confidence * 100.0
        let percent = String(format: "%.2f%%", confidence)
        
        // Print the result as Instrument: percentage confidence.
        print("\(classification.identifier): \(percent) confidence.\n")
        
        DispatchQueue.main.async {
            self.vc?.lblPrediction.text = classification.identifier.localizedCapitalized
            print("\(classification.identifier) \(percent)")
            let term = classification.identifier.lowercased()
            if (term == "hungry") {
                self.vc?.lblTip.text = "Feed the baby!"
                self.vc?.lblPrediction.textColor = .systemCyan
            }else if (term == "discomfort") {
                self.vc?.lblTip.text = "Check baby's diapers!"
                self.vc?.lblPrediction.textColor = .systemOrange
            }else if (term == "tired") {
                self.vc?.lblTip.text = "Your baby needs a nap!"
                self.vc?.lblPrediction.textColor = .systemGreen
            }else if (term == "burping") {
                self.vc?.lblTip.text = "Help your baby to release air from the stomach through the mouth!"
                self.vc?.lblPrediction.textColor = .systemOrange
            }else if (term == "belly_pain") {
                self.vc?.lblTip.text = "Your baby is suffering from a belly pain!"
                self.vc?.lblPrediction.textColor = .systemRed
            }
        }
    }
    
    func request(_ request: SNRequest, didFailWithError error: Error) {
        print("The the analysis failed: \(error.localizedDescription)")
    }
    
    func requestDidComplete(_ request: SNRequest) {
        print("The request completed successfully!")
    }
}
