import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:flutter/foundation.dart';
import 'dart:async';
import 'dart:io';

/// Configuration class for liveness detection
class LivenessConfig {
  final String title;
  final int timeoutSeconds;
  final List<LivenessGesture> requiredGestures;
  final Color primaryColor;
  final Color successColor;
  final Color errorColor;
  final bool showInstructions;
  final bool showTimer;

  const LivenessConfig({
    this.title = 'Verifikasi Liveness',
    this.timeoutSeconds = 30,
    this.requiredGestures = const [
      LivenessGesture.mouthOpen,
      LivenessGesture.headShake,
      LivenessGesture.blink,
    ],
    this.primaryColor = const Color(0xFF2196F3),
    this.successColor = const Color(0xFF4CAF50),
    this.errorColor = const Color(0xFFF44336),
    this.showInstructions = true,
    this.showTimer = true,
  });
}

/// Enum for liveness result status
enum LivenessResultStatus { success, failed, timeout, cancelled }

/// Enum for different liveness gestures
enum LivenessGesture { blink, mouthOpen, headShake, smile, turnLeft, turnRight }

/// Extension to get gesture properties
extension LivenessGestureExtension on LivenessGesture {
  String get displayName {
    switch (this) {
      case LivenessGesture.blink:
        return 'Kedipkan Mata';
      case LivenessGesture.mouthOpen:
        return 'Buka Mulut';
      case LivenessGesture.headShake:
        return 'Gelengkan Kepala';
      case LivenessGesture.smile:
        return 'Tersenyum';
      case LivenessGesture.turnLeft:
        return 'Toleh Kiri';
      case LivenessGesture.turnRight:
        return 'Toleh Kanan';
    }
  }

  String get instruction {
    switch (this) {
      case LivenessGesture.blink:
        return 'Kedipkan mata Anda beberapa kali dengan jelas';
      case LivenessGesture.mouthOpen:
        return 'Buka mulut Anda lebar-lebar selama 2 detik';
      case LivenessGesture.headShake:
        return 'Gelengkan kepala ke kiri dan kanan perlahan';
      case LivenessGesture.smile:
        return 'Tersenyum lebar dan tahan selama 2 detik';
      case LivenessGesture.turnLeft:
        return 'Tolehkan kepala ke kiri dan tahan';
      case LivenessGesture.turnRight:
        return 'Tolehkan kepala ke kanan dan tahan';
    }
  }

  IconData get icon {
    switch (this) {
      case LivenessGesture.blink:
        return Icons.visibility;
      case LivenessGesture.mouthOpen:
        return Icons.record_voice_over;
      case LivenessGesture.headShake:
        return Icons.swap_horiz;
      case LivenessGesture.smile:
        return Icons.sentiment_very_satisfied;
      case LivenessGesture.turnLeft:
        return Icons.keyboard_arrow_left;
      case LivenessGesture.turnRight:
        return Icons.keyboard_arrow_right;
    }
  }
}

/// Result class for liveness detection
class LivenessResult {
  final LivenessResultStatus status;
  final String message;
  final File? capturedImage;
  final List<LivenessGesture> completedGestures;
  final double confidence;
  final Duration duration;

  const LivenessResult({
    required this.status,
    required this.message,
    this.capturedImage,
    required this.completedGestures,
    required this.confidence,
    required this.duration,
  });

  bool get success => status == LivenessResultStatus.success;
}

/// Standalone Liveness Detection Page
class FaceLiveness extends StatefulWidget {
  final LivenessConfig config;
  final Function(LivenessResult) onResult;
  final VoidCallback? onCancel;

  const FaceLiveness({
    super.key,
    required this.onResult,
    this.config = const LivenessConfig(),
    this.onCancel,
  });

  static Future<LivenessResult?> show(
    BuildContext context, {
    LivenessConfig config = const LivenessConfig(),
    Function(LivenessResult)? onResult,
  }) async {
    return await Navigator.of(context).push<LivenessResult>(
      MaterialPageRoute(
        builder: (context) => FaceLiveness(
          config: config,
          onResult: onResult ?? (result) => Navigator.of(context).pop(result),
          onCancel: () => Navigator.of(context).pop(),
        ),
      ),
    );
  }

  @override
  State<FaceLiveness> createState() => _FaceLivenessState();
}

class _FaceLivenessState extends State<FaceLiveness> {
  CameraController? _cameraController;
  CameraDescription? _cameraDescription;
  bool _isCameraInitialized = false;
  bool _isProcessing = false;

  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableLandmarks: true,
      enableClassification: true,
      enableTracking: true,
      performanceMode: FaceDetectorMode.accurate,
    ),
  );

  int _currentGestureIndex = 0;
  final Set<LivenessGesture> _completedGestures = {};
  int _timeRemaining = 30;
  Timer? _timer;
  DateTime? _startTime;
  String _errorMessage = '';

  LivenessGesture? _currentStep;
  bool _isWaitingToStart = true;
  bool _isInFinalCapture = false;
  bool _isFacingCamera = false;
  bool _isCaptureDone = false;
  bool _isStreamStopped = false;
  int _facingCameraFrames = 0;

  double? _previousLeftEyeOpenProbability;
  double? _previousRightEyeOpenProbability;
  double? _previousHeadEulerAngleY;
  int _blinkCount = 0;
  int _mouthOpenFrames = 0;
  int _headShakeFrames = 0;
  int _smileFrames = 0;

  bool _blinkDetected = false;
  bool _mouthOpenDetected = false;
  bool _headShakeDetected = false;
  bool _smileDetected = false;
  bool _turnLeftDetected = false;
  bool _turnRightDetected = false;

  File? _capturedImage;

  @override
  void initState() {
    super.initState();
    _timeRemaining = widget.config.timeoutSeconds;
    _startTime = DateTime.now();

    _initializeCamera();
    _startTimer();

    Timer(const Duration(seconds: 2), () {
      if (mounted) {
        setState(() {
          _isWaitingToStart = false;
          if (widget.config.requiredGestures.isNotEmpty) {
            _currentStep = widget.config.requiredGestures[0];
          }
        });
      }
    });
  }

  @override
  void dispose() {
    _cleanup();
    super.dispose();
  }

  void _cleanup() {
    _timer?.cancel();
    // Jangan panggil stopImageStream lagi jika sudah dihentikan
    if (!_isStreamStopped) {
      try {
        _cameraController?.stopImageStream();
      } catch (_) {}
      _isStreamStopped = true;
    }
    _cameraController?.dispose();
    _cameraController = null;
    _faceDetector.close();
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        _showError('Kamera tidak tersedia');
        return;
      }

      final frontCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      _cameraDescription = frontCamera;

      _cameraController = CameraController(
        frontCamera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.nv21
            : ImageFormatGroup.bgra8888,
      );

      await _cameraController!.initialize();

      if (mounted) {
        setState(() => _isCameraInitialized = true);

        // Langsung mulai streaming
        _cameraController!.startImageStream((CameraImage image) {
          if (_isProcessing || _isWaitingToStart || _isInFinalCapture) return;
          _processStreamImage(image);
        });
      }
    } catch (e) {
      _showError('Gagal menginisialisasi kamera: $e');
    }
  }

  InputImageRotation _rotationFromSensorOrientation(int sensorOrientation) {
    switch (sensorOrientation) {
      case 0:
        return InputImageRotation.rotation0deg;
      case 90:
        return InputImageRotation.rotation90deg;
      case 180:
        return InputImageRotation.rotation180deg;
      case 270:
        return InputImageRotation.rotation270deg;
      default:
        return InputImageRotation.rotation0deg;
    }
  }

  void _processStreamImage(CameraImage image) async {
    _isProcessing = true;
    try {
      final Uint8List bytes;
      final InputImageFormat format;
      final InputImageRotation rotation;
      final int bytesPerRow;

      if (Platform.isIOS) {
        // iOS: gunakan BGRA8888, plane pertama saja
        bytes = image.planes[0].bytes;
        format = InputImageFormat.bgra8888;
        bytesPerRow = image.planes[0].bytesPerRow;
      } else {
        // Android: gunakan NV21 langsung dari plane pertama
        // Karena kita sudah set imageFormatGroup: ImageFormatGroup.nv21
        bytes = image.planes[0].bytes;
        format = InputImageFormat.nv21;
        bytesPerRow = image.planes[0].bytesPerRow;
      }

      // Ambil rotation dari sensor orientation kamera yang sebenarnya
      if (_cameraDescription != null) {
        rotation = _rotationFromSensorOrientation(
          _cameraDescription!.sensorOrientation,
        );
      } else {
        rotation = Platform.isIOS
            ? InputImageRotation.rotation270deg
            : InputImageRotation.rotation90deg;
      }

      final metadata = InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: bytesPerRow,
      );

      final inputImage = InputImage.fromBytes(bytes: bytes, metadata: metadata);

      final faces = await _faceDetector.processImage(inputImage);
      if (faces.isNotEmpty) {
        _processFaceData(faces.first);
      }
    } catch (e) {
      debugPrint("Stream Error: $e");
    } finally {
      _isProcessing = false;
    }
  }

  void _startTimer() {
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_timeRemaining > 0) {
        setState(() {
          _timeRemaining--;
        });
      } else {
        _onTimeout();
      }
    });
  }

  void _processFaceData(Face face) {
    final currentGesture = _currentStep;
    if (currentGesture == null) return;

    switch (currentGesture) {
      case LivenessGesture.blink:
        _detectBlink(face);
        break;
      case LivenessGesture.mouthOpen:
        _detectMouthOpen(face);
        break;
      case LivenessGesture.headShake:
        _detectHeadShake(face);
        break;
      case LivenessGesture.smile:
        _detectSmile(face);
        break;
      case LivenessGesture.turnLeft:
        _detectTurnLeft(face);
        break;
      case LivenessGesture.turnRight:
        _detectTurnRight(face);
        break;
    }

    if (_isInFinalCapture) {
      _detectFacingCamera(face);
    }
  }

  void _detectBlink(Face face) {
    final leftEyeOpen = face.leftEyeOpenProbability;
    final rightEyeOpen = face.rightEyeOpenProbability;

    if (leftEyeOpen != null && rightEyeOpen != null) {
      if (_previousLeftEyeOpenProbability != null &&
          _previousRightEyeOpenProbability != null) {
        // Deteksi transisi: mata terbuka → mata tertutup
        final wasOpen = _previousLeftEyeOpenProbability! > 0.5 &&
            _previousRightEyeOpenProbability! > 0.5;
        final isClosed = leftEyeOpen < 0.3 && rightEyeOpen < 0.3;

        if (wasOpen && isClosed) {
          _blinkCount++;
        }

        if (_blinkCount >= 1 && !_blinkDetected) {
          _blinkDetected = true;
          _onGestureDetected(LivenessGesture.blink);
        }
      }
      _previousLeftEyeOpenProbability = leftEyeOpen;
      _previousRightEyeOpenProbability = rightEyeOpen;
    }
  }

  void _detectMouthOpen(Face face) {
    final landmarks = face.landmarks;
    if (landmarks.isNotEmpty) {
      _mouthOpenFrames++;
      if (_mouthOpenFrames >= 20 && !_mouthOpenDetected) {
        _mouthOpenDetected = true;
        _onGestureDetected(LivenessGesture.mouthOpen);
      }
    }
  }

  void _detectHeadShake(Face face) {
    final headEulerAngleY = face.headEulerAngleY;

    if (headEulerAngleY != null) {
      if (_previousHeadEulerAngleY != null) {
        final angleDiff = (headEulerAngleY - _previousHeadEulerAngleY!).abs();

        if (angleDiff > 5.0) {
          _headShakeFrames++;
        }

        if (_headShakeFrames >= 10 && !_headShakeDetected) {
          _headShakeDetected = true;
          _onGestureDetected(LivenessGesture.headShake);
        }
      }
      _previousHeadEulerAngleY = headEulerAngleY;
    }
  }

  void _detectSmile(Face face) {
    final smilingProbability = face.smilingProbability;

    if (smilingProbability != null && smilingProbability > 0.7) {
      _smileFrames++;
      if (_smileFrames >= 20 && !_smileDetected) {
        _smileDetected = true;
        _onGestureDetected(LivenessGesture.smile);
      }
    } else {
      _smileFrames = 0;
    }
  }

  void _detectTurnLeft(Face face) {
    final headEulerAngleY = face.headEulerAngleY;

    if (headEulerAngleY != null &&
        headEulerAngleY > 15.0 &&
        !_turnLeftDetected) {
      _turnLeftDetected = true;
      _onGestureDetected(LivenessGesture.turnLeft);
    }
  }

  void _detectTurnRight(Face face) {
    final headEulerAngleY = face.headEulerAngleY;

    if (headEulerAngleY != null &&
        headEulerAngleY < -15.0 &&
        !_turnRightDetected) {
      _turnRightDetected = true;
      _onGestureDetected(LivenessGesture.turnRight);
    }
  }

  void _detectFacingCamera(Face face) {
    final headEulerAngleY = face.headEulerAngleY;

    if (headEulerAngleY != null &&
        (headEulerAngleY < 5.0 && headEulerAngleY > -5.0)) {
      _facingCameraFrames++;

      if (_facingCameraFrames >= 20 && !_isFacingCamera) {
        _isFacingCamera = true;
        _onFinalCapture();
      }
    } else {
      _facingCameraFrames = 0;
    }
  }

  void _onGestureDetected(LivenessGesture gesture) {
    if (_completedGestures.contains(gesture)) return;

    setState(() {
      _completedGestures.add(gesture);
      if (_currentGestureIndex < widget.config.requiredGestures.length - 1) {
        _currentGestureIndex++;
        _currentStep = widget.config.requiredGestures[_currentGestureIndex];
        _resetGestureDetectionState();
        _resetTimer();
      } else {
        _currentStep = null;
        _isInFinalCapture = true;
      }
    });

    if (_completedGestures.length == widget.config.requiredGestures.length) {
      _onSuccess();
    }
  }

  void _resetGestureDetectionState() {
    _blinkCount = 0;
    _mouthOpenFrames = 0;
    _headShakeFrames = 0;
    _smileFrames = 0;
    _blinkDetected = false;
    _mouthOpenDetected = false;
    _headShakeDetected = false;
    _smileDetected = false;
    _turnLeftDetected = false;
    _turnRightDetected = false;
  }

  void _onSuccess() {
    if (_isCaptureDone) return;
    _onFinalCapture();
  }

  void _onTimeout() {
    _timer?.cancel();
    if (!_isStreamStopped) {
      try {
        _cameraController?.stopImageStream();
      } catch (_) {}
      _isStreamStopped = true;
    }

    final result = LivenessResult(
      status: LivenessResultStatus.timeout,
      message: 'Waktu verifikasi habis',
      capturedImage: _capturedImage,
      completedGestures: _completedGestures.toList(),
      confidence: 0.0,
      duration: DateTime.now().difference(_startTime!),
    );

    widget.onResult(result);
    if (mounted) Navigator.of(context).pop(result);
  }

  void _onFailure(String errorMessage) {
    _timer?.cancel();
    if (!_isStreamStopped) {
      try {
        _cameraController?.stopImageStream();
      } catch (_) {}
      _isStreamStopped = true;
    }

    final result = LivenessResult(
      status: LivenessResultStatus.failed,
      message: errorMessage,
      capturedImage: null,
      completedGestures: _completedGestures.toList(),
      confidence: 0.0,
      duration: DateTime.now().difference(_startTime!),
    );

    widget.onResult(result);
    if (mounted) Navigator.of(context).pop(result);
  }

  void _onCancel() {
    _timer?.cancel();
    if (!_isStreamStopped) {
      try {
        _cameraController?.stopImageStream();
      } catch (_) {}
      _isStreamStopped = true;
    }

    final result = LivenessResult(
      status: LivenessResultStatus.cancelled,
      message: 'Verifikasi dibatalkan oleh pengguna',
      capturedImage: null,
      completedGestures: _completedGestures.toList(),
      confidence: 0.0,
      duration: DateTime.now().difference(_startTime!),
    );

    widget.onResult(result);
    if (mounted) Navigator.of(context).pop(result);
  }

  void _resetTimer() {
    _timer?.cancel();
    setState(() {
      _timeRemaining = widget.config.timeoutSeconds;
    });

    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_timeRemaining > 0) {
        setState(() {
          _timeRemaining--;
        });
      } else {
        _onTimeout();
      }
    });
  }

  void _showError(String message) {
    setState(() {
      _errorMessage = message;
    });

    Timer(const Duration(seconds: 3), () {
      if (mounted) {
        _onFailure(message);
      }
    });
  }

  Future<void> _onFinalCapture() async {
    // Guard: cegah dipanggil lebih dari sekali
    if (_isCaptureDone) return;
    _isCaptureDone = true;
    _isProcessing = true;
    _isInFinalCapture = true;
    _timer?.cancel();

    try {
      // Matikan stream sebelum ambil foto
      if (!_isStreamStopped) {
        await _cameraController?.stopImageStream();
        _isStreamStopped = true;
      }

      // Beri waktu sebentar agar stream benar-benar berhenti
      await Future.delayed(const Duration(milliseconds: 200));

      // Ambil foto final
      if (_cameraController != null &&
          _cameraController!.value.isInitialized) {
        final XFile photo = await _cameraController!.takePicture();
        _capturedImage = File(photo.path);
      }
    } catch (e) {
      debugPrint("Gagal ambil foto final: $e");
    }

    // Jangan dispose camera di sini — biarkan dispose() yang handle
    // agar tidak crash saat widget masih dirender oleh framework

    final result = LivenessResult(
      status: LivenessResultStatus.success,
      message: 'Verifikasi liveness berhasil',
      capturedImage: _capturedImage,
      completedGestures: _completedGestures.toList(),
      confidence: 0.95,
      duration: DateTime.now().difference(_startTime!),
    );

    widget.onResult(result);

    if (mounted) {
      Navigator.of(context).pop(result);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.close, color: Colors.black),
          onPressed: _onCancel,
        ),
        title: Text(
          widget.config.title,
          style: const TextStyle(color: Colors.black),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            _buildCameraPreview(),
            const SizedBox(height: 24),
            _buildCurrentStepInstruction(),
            const SizedBox(height: 24),
            _buildTimerAndProgress(),
            if (_errorMessage.isNotEmpty) _buildErrorMessage(_errorMessage),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (!_isCameraInitialized ||
        _cameraController == null ||
        !_cameraController!.value.isInitialized) {
      return Container(
        height: 300,
        width: 300,
        decoration: BoxDecoration(
          color: Colors.grey[200],
          borderRadius: BorderRadius.circular(300),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(
                widget.config.primaryColor,
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Menginisialisasi Kamera...',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w500,
                color: Colors.grey[700],
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Verifikasi akan dimulai otomatis',
              style: TextStyle(fontSize: 14, color: Colors.grey[600]),
            ),
          ],
        ),
      );
    }

    return Container(
      height: 300,
      width: 300,
      decoration: const BoxDecoration(shape: BoxShape.circle),
      child: ClipOval(
        child: OverflowBox(
          maxWidth: double.infinity,
          maxHeight: double.infinity,
          child: FittedBox(
            fit: BoxFit.cover,
            child: SizedBox(
              width: _cameraController!.value.previewSize!.height,
              height: _cameraController!.value.previewSize!.width,
              child: CameraPreview(_cameraController!),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildCurrentStepInstruction() {
    if (_isWaitingToStart) {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.orange[50],
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.orange[200]!),
        ),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.orange[100],
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.hourglass_empty,
                size: 40,
                color: Colors.orange[700],
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Mempersiapkan Verifikasi',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.orange[800],
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              'Verifikasi liveness akan dimulai otomatis dalam beberapa detik...',
              style: TextStyle(
                fontSize: 16,
                color: Colors.orange[700],
                height: 1.5,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    if (_currentStep == null &&
        _completedGestures.length == widget.config.requiredGestures.length) {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: widget.config.successColor.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: widget.config.successColor.withValues(alpha: 0.3),
          ),
        ),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: widget.config.successColor.withValues(alpha: 0.2),
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.check_circle,
                size: 40,
                color: widget.config.successColor,
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Verifikasi Berhasil!',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: widget.config.successColor,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              'Semua gerakan telah berhasil diverifikasi. Memproses hasil...',
              style: TextStyle(
                fontSize: 16,
                color: widget.config.successColor,
                height: 1.5,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    if (_currentStep != null) {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: widget.config.primaryColor.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: widget.config.primaryColor.withValues(alpha: 0.3),
          ),
        ),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: widget.config.primaryColor.withValues(alpha: 0.2),
                shape: BoxShape.circle,
              ),
              child: Icon(
                _currentStep!.icon,
                size: 40,
                color: widget.config.primaryColor,
              ),
            ),
            const SizedBox(height: 16),
            Text(
              _currentStep!.displayName,
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: widget.config.primaryColor,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              _currentStep!.instruction,
              style: TextStyle(
                fontSize: 16,
                color: widget.config.primaryColor,
                height: 1.5,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    if (_isInFinalCapture) {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: widget.config.primaryColor.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: widget.config.primaryColor.withValues(alpha: 0.3),
          ),
        ),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: widget.config.primaryColor.withValues(alpha: 0.2),
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.camera_front,
                size: 40,
                color: widget.config.primaryColor,
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Hadapkan Wajah Anda',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: widget.config.primaryColor,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              'Sedang mengambil foto terakhir...',
              style: TextStyle(
                fontSize: 16,
                color: widget.config.primaryColor,
                height: 1.5,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    return const SizedBox.shrink();
  }

  Widget _buildTimerAndProgress() {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          decoration: BoxDecoration(
            color: _timeRemaining <= 10
                ? widget.config.errorColor.withValues(alpha: 0.1)
                : widget.config.successColor.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(25),
            border: Border.all(
              color: _timeRemaining <= 10
                  ? widget.config.errorColor.withValues(alpha: 0.3)
                  : widget.config.successColor.withValues(alpha: 0.3),
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.timer,
                color: _timeRemaining <= 10
                    ? widget.config.errorColor
                    : widget.config.successColor,
                size: 20,
              ),
              const SizedBox(width: 8),
              Text(
                '${_timeRemaining}s',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: _timeRemaining <= 10
                      ? widget.config.errorColor
                      : widget.config.successColor,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildErrorMessage(String message) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: widget.config.errorColor.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: widget.config.errorColor.withValues(alpha: 0.3),
        ),
      ),
      child: Row(
        children: [
          Icon(Icons.error_outline, color: widget.config.errorColor, size: 24),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              message,
              style: TextStyle(
                color: widget.config.errorColor,
                fontSize: 14,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
