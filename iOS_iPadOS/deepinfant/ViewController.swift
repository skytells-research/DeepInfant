//
//  ViewController.swift
//  deepinfant
//
//  Created by Hazem Ali on 2/10/22.
//


import UIKit
import CoreML
import SoundAnalysis
import AVFoundation

class ViewController: UIViewController {
    
    @IBOutlet weak var btnAction: UIButton!
    @IBOutlet weak var lblPrediction: UILabel!
    @IBOutlet weak var lblTip: UILabel!
    
    
    // Audio Engine
    var audioEngine = AVAudioEngine()
    
    // Streaming Audio Analyzer
    var streamAnalyzer: SNAudioStreamAnalyzer!
    
    // Serial dispatch queue used to analyze incoming audio buffers.
    let analysisQueue = DispatchQueue(label: "com.skytells.research.AnalysisQueue")

    var resultsObserver: ResultsObserver!
    
    // Instantiate the ML Model
    lazy var mlClassifier = DeepInfant_VGGish()
   
    
    override func viewDidLoad() {
        super.viewDidLoad()
        //self.startAudioEngine()
        
        // Get the native audio format of the engine's input bus.
        let inputFormat = self.audioEngine.inputNode.inputFormat(forBus: 0)

        // Create a new stream analyzer.
        streamAnalyzer = SNAudioStreamAnalyzer(format: inputFormat)

        // Create a new observer that will be notified of analysis results.
        // Keep a strong reference to this object.
        resultsObserver = ResultsObserver()
        resultsObserver.vc = self

        do {
            var model: MLModel = try MLModel.init(contentsOf: Bundle.main.url(forResource: "DeepInfant_VGGish", withExtension: "mlmodelc")!)
            // Prepare a new request for the trained model.
            let request = try SNClassifySoundRequest(mlModel: model)
            try streamAnalyzer.add(request, withObserver: resultsObserver)
        } catch {
            print("Unable to prepare request: \(error.localizedDescription)")
            return
        }

        // Install an audio tap on the audio engine's input node.
        self.audioEngine.inputNode.installTap(onBus: 0,
                                         bufferSize: 8192, // 8k buffer
        format: inputFormat) { buffer, time in
            // Analyze the current audio buffer.
            self.analysisQueue.async {
                self.streamAnalyzer.analyze(buffer, atAudioFramePosition: time.sampleTime)
            }
        }
       // self.startAudioEngine()
       // print("Result....")
    }
    
    @IBAction func start(_ sender: Any) {
        if (btnAction.titleLabel!.text!.lowercased() == "start") {
           
            startAudioEngine()
        }else {
           
            stopEngine()
        }
       
    }
    
    func stopEngine() {
        self.audioEngine.stop()
        btnAction.setTitle("Start", for: [])
        lblPrediction.text = "Ready"
        lblTip.text = "Tap Start and move closer to the baby.."
        lblPrediction.textColor = .label
        btnAction.backgroundColor = .systemBlue
    }
    // Function to Start Audio Engine for Recording Audio
    func startAudioEngine() {
        do {
            // Start the stream of audio data.
            try self.audioEngine.start()
            btnAction.setTitle("Listening..", for: [])
            lblTip.text = "Analyzing Sound Buffers.."
            btnAction.backgroundColor = .systemRed
        } catch {
            print("Unable to start AVAudioEngine: \(error.localizedDescription)")
        }
    }
}
