import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

// ============================================================================
// ENTRY POINT
// ============================================================================
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);
  await SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  await WakelockPlus.enable();
  await Permission.camera.request();
  await StorageService.instance.init();
  await PoseTrackerService.instance.initialize();
  runApp(const RompMotionApp());
}

class RompMotionApp extends StatelessWidget {
  const RompMotionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RompMotion',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      builder: (context, child) => HandCursorOverlay(child: child!),
      home: const CalibrationScreen(),
    );
  }
}

// ============================================================================
// DATA MODELS
// ============================================================================
enum FramingStatus { noPlayer, headCutOff, feetCutOff, optimal }

enum GameMode { solo, together, takeTurns }

class TrackedPlayer {
  final int playerIndex;
  final Map<PoseLandmarkType, Offset> landmarks;
  final Map<PoseLandmarkType, double> likelihoods;

  TrackedPlayer({
    required this.playerIndex,
    required this.landmarks,
    required this.likelihoods,
  });

  Offset? getLandmark(PoseLandmarkType type) => landmarks[type];

  bool isVisible(PoseLandmarkType type, {double threshold = 0.50}) {
    return (likelihoods[type] ?? 0.0) >= threshold;
  }

  FramingStatus get framingStatus {
    final nose = getLandmark(PoseLandmarkType.nose);
    final leftAnkle = getLandmark(PoseLandmarkType.leftAnkle);
    final rightAnkle = getLandmark(PoseLandmarkType.rightAnkle);

    if (nose == null || !isVisible(PoseLandmarkType.nose)) {
      return FramingStatus.noPlayer;
    }
    if (nose.dy < 0.06) {
      return FramingStatus.headCutOff;
    }
    final bool anklesVisible =
        isVisible(PoseLandmarkType.leftAnkle, threshold: 0.40) ||
        isVisible(PoseLandmarkType.rightAnkle, threshold: 0.40);
    final double lowestAnkleY = [
      if (leftAnkle != null) leftAnkle.dy,
      if (rightAnkle != null) rightAnkle.dy,
    ].fold(0.0, (a, b) => a > b ? a : b);

    if (!anklesVisible || lowestAnkleY > 0.96) {
      return FramingStatus.feetCutOff;
    }
    return FramingStatus.optimal;
  }

  bool isFistClosed({required bool isRightHand}) {
    final wrist = getLandmark(
        isRightHand ? PoseLandmarkType.rightWrist : PoseLandmarkType.leftWrist);
    final index = getLandmark(
        isRightHand ? PoseLandmarkType.rightIndex : PoseLandmarkType.leftIndex);
    final pinky = getLandmark(
        isRightHand ? PoseLandmarkType.rightPinky : PoseLandmarkType.leftPinky);
    if (wrist == null || index == null || pinky == null) return false;
    return (index - wrist).distance < 0.032 && (index - pinky).distance < 0.022;
  }
}

class PoseFrameResult {
  final List<TrackedPlayer> players;
  final double averageFrameLuminance;
  final int timestampMs;

  PoseFrameResult({
    required this.players,
    required this.averageFrameLuminance,
    required this.timestampMs,
  });
}

class GameProgress {
  int xp;
  int streakDays;
  String lastPlayedDate;
  int latencyOffsetMs;
  Map<String, int> bestStars;

  GameProgress({
    this.xp = 0,
    this.streakDays = 1,
    this.lastPlayedDate = '',
    this.latencyOffsetMs = 110,
    Map<String, int>? bestStars,
  }) : bestStars = bestStars ?? {'slice': 0, 'sprint': 0, 'pulse': 0};

  int get level => (xp ~/ 500) + 1;
}

// ============================================================================
// SERVICES (POSE TRACKING, LOCAL SAVE, SOUND)
// ============================================================================
class PoseTrackerService extends ChangeNotifier {
  static final PoseTrackerService instance = PoseTrackerService._internal();
  PoseTrackerService._internal();

  CameraController? _cameraController;
  PoseDetector? _poseDetector;
  bool _isProcessing = false;
  bool _isInitialized = false;

  final StreamController<PoseFrameResult> _poseStreamController =
      StreamController<PoseFrameResult>.broadcast();

  Stream<PoseFrameResult> get poseStream => _poseStreamController.stream;
  CameraController? get cameraController => _cameraController;

  Future<void> initialize() async {
    if (_isInitialized) return;
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    final frontCamera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    _cameraController = CameraController(
      frontCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.nv21
          : ImageFormatGroup.bgra8888,
    );

    await _cameraController!.initialize();
    await _cameraController!
        .lockCaptureOrientation(DeviceOrientation.landscapeLeft);

    _poseDetector = PoseDetector(
      options: PoseDetectorOptions(
        mode: PoseDetectionMode.stream,
        model: PoseDetectionModel.base,
      ),
    );

    _isInitialized = true;
    notifyListeners();
    await _cameraController!.startImageStream(_processCameraImage);
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isProcessing || _poseDetector == null) return;
    _isProcessing = true;

    try {
      final int nowMs = DateTime.now().millisecondsSinceEpoch;
      final double luminance = _calculateAverageLuminance(image);
      final inputImage = _inputImageFromCameraImage(image);
      if (inputImage == null) {
        _isProcessing = false;
        return;
      }

      final poses = await _poseDetector!.processImage(inputImage);
      final double imgW = image.width.toDouble();
      final double imgH = image.height.toDouble();
      final List<TrackedPlayer> trackedPlayers = [];

      poses.sort((a, b) {
        final ax = 1.0 - ((a.landmarks[PoseLandmarkType.nose]?.x ?? 0) / imgW);
        final bx = 1.0 - ((b.landmarks[PoseLandmarkType.nose]?.x ?? 0) / imgW);
        return ax.compareTo(bx);
      });

      for (int i = 0; i < poses.length && i < 2; i++) {
        final pose = poses[i];
        final Map<PoseLandmarkType, Offset> normalized = {};
        final Map<PoseLandmarkType, double> likelihoods = {};

        pose.landmarks.forEach((type, landmark) {
          final double normX = (1.0 - (landmark.x / imgW)).clamp(0.0, 1.0);
          final double normY = (landmark.y / imgH).clamp(0.0, 1.0);
          normalized[type] = Offset(normX, normY);
          likelihoods[type] = landmark.likelihood;
        });

        trackedPlayers.add(TrackedPlayer(
          playerIndex: i,
          landmarks: normalized,
          likelihoods: likelihoods,
        ));
      }

      _poseStreamController.add(PoseFrameResult(
        players: trackedPlayers,
        averageFrameLuminance: luminance,
        timestampMs: nowMs,
      ));
    } catch (_) {
    } finally {
      _isProcessing = false;
    }
  }

  double _calculateAverageLuminance(CameraImage image) {
    final Uint8List yPlane = image.planes[0].bytes;
    int sum = 0;
    int count = 0;
    for (int i = 0; i < yPlane.length; i += 32) {
      sum += yPlane[i];
      count++;
    }
    return count == 0 ? 0.0 : sum / count;
  }

  InputImage? _inputImageFromCameraImage(CameraImage image) {
    if (_cameraController == null) return null;
    final camera = _cameraController!.description;
    final rotation =
        InputImageRotationValue.fromRawValue(camera.sensorOrientation) ??
            InputImageRotation.rotation0deg;
    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null) return null;

    final plane = image.planes.first;
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }
}

class StorageService {
  static final StorageService instance = StorageService._internal();
  StorageService._internal();

  late SharedPreferences _prefs;
  GameProgress progress = GameProgress();

  Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
    final today = DateTime.now().toIso8601String().substring(0, 10);
    final savedDate = _prefs.getString('lastPlayedDate') ?? '';
    int streak = _prefs.getInt('streakDays') ?? 1;

    if (savedDate.isNotEmpty && savedDate != today) {
      final yesterday = DateTime.now()
          .subtract(const Duration(days: 1))
          .toIso8601String()
          .substring(0, 10);
      streak = (savedDate == yesterday) ? streak + 1 : 1;
    }

    final starsJson = _prefs.getString('bestStars');
    final Map<String, int> stars = starsJson != null
        ? Map<String, int>.from(jsonDecode(starsJson))
        : {'slice': 0, 'sprint': 0, 'pulse': 0};

    progress = GameProgress(
      xp: _prefs.getInt('xp') ?? 0,
      streakDays: streak,
      lastPlayedDate: today,
      latencyOffsetMs: _prefs.getInt('latencyOffsetMs') ?? 110,
      bestStars: stars,
    );
    await save();
  }

  Future<void> recordGameResult({
    required String gameKey,
    required int earnedXp,
    required int stars,
  }) async {
    progress.xp += earnedXp;
    if (stars > (progress.bestStars[gameKey] ?? 0)) {
      progress.bestStars[gameKey] = stars;
    }
    await save();
  }

  Future<void> setLatencyOffset(int ms) async {
    progress.latencyOffsetMs = ms;
    await save();
  }

  Future<void> save() async {
    await _prefs.setInt('xp', progress.xp);
    await _prefs.setInt('streakDays', progress.streakDays);
    await _prefs.setString('lastPlayedDate', progress.lastPlayedDate);
    await _prefs.setInt('latencyOffsetMs', progress.latencyOffsetMs);
    await _prefs.setString('bestStars', jsonEncode(progress.bestStars));
  }
}

// ============================================================================
// TOUCH-FREE HAND CURSOR & DWELL BUTTON WIDGETS
// ============================================================================
class HandCursorRegistry {
  static final List<DwellTargetState> targets = [];
  static void register(DwellTargetState target) => targets.add(target);
  static void unregister(DwellTargetState target) => targets.remove(target);
}

mixin DwellTargetState<T extends StatefulWidget> on State<T> {
  final GlobalKey targetKey = GlobalKey();
  VoidCallback get onTriggered;

  Rect? get globalRect {
    final renderBox =
        targetKey.currentContext?.findRenderObject() as RenderBox?;
    if (renderBox == null || !renderBox.hasSize) return null;
    final offset = renderBox.localToGlobal(Offset.zero);
    return offset & renderBox.size;
  }

  @override
  void initState() {
    super.initState();
    HandCursorRegistry.register(this);
  }

  @override
  void dispose() {
    HandCursorRegistry.unregister(this);
    super.dispose();
  }
}

class HandCursorOverlay extends StatefulWidget {
  final Widget child;
  const HandCursorOverlay({super.key, required this.child});

  @override
  State<HandCursorOverlay> createState() => _HandCursorOverlayState();
}

class _HandCursorOverlayState extends State<HandCursorOverlay> {
  StreamSubscription<PoseFrameResult>? _sub;
  Offset _smoothedCursor = const Offset(0.5, 0.5);
  bool _hasHand = false;
  double _dwellProgress = 0.0;
  DwellTargetState? _hoveredTarget;
  DateTime _lastTick = DateTime.now();
  bool _fistCooldown = false;

  @override
  void initState() {
    super.initState();
    _sub = PoseTrackerService.instance.poseStream.listen(_onPoseUpdate);
  }

  void _onPoseUpdate(PoseFrameResult frame) {
    if (!mounted) return;
    final now = DateTime.now();
    final dt = now.difference(_lastTick).inMilliseconds / 1000.0;
    _lastTick = now;

    if (frame.players.isEmpty) {
      if (_hasHand) setState(() => _hasHand = false);
      return;
    }

    final player = frame.players.first;
    final rWrist = player.getLandmark(PoseLandmarkType.rightWrist);
    final lWrist = player.getLandmark(PoseLandmarkType.leftWrist);

    Offset? activeWrist;
    bool usingRightHand = true;

    if (rWrist != null && lWrist != null) {
      if (rWrist.dy <= lWrist.dy) {
        activeWrist = rWrist;
        usingRightHand = true;
      } else {
        activeWrist = lWrist;
        usingRightHand = false;
      }
    } else {
      activeWrist = rWrist ?? lWrist;
      usingRightHand = rWrist != null;
    }

    if (activeWrist == null) {
      setState(() => _hasHand = false);
      return;
    }

    final double mappedX = ((activeWrist.dx - 0.15) / 0.70).clamp(0.0, 1.0);
    final double mappedY = ((activeWrist.dy - 0.15) / 0.60).clamp(0.0, 1.0);
    _smoothedCursor = Offset.lerp(_smoothedCursor, Offset(mappedX, mappedY), 0.35)!;
    _hasHand = true;

    final size = MediaQuery.of(context).size;
    final screenPoint = Offset(
      _smoothedCursor.dx * size.width,
      _smoothedCursor.dy * size.height,
    );

    DwellTargetState? hit;
    for (final target in HandCursorRegistry.targets) {
      final rect = target.globalRect;
      if (rect != null && rect.contains(screenPoint)) {
        hit = target;
        break;
      }
    }

    final bool fistClosed = player.isFistClosed(isRightHand: usingRightHand);

    if (hit != null) {
      if (_hoveredTarget != hit) {
        _hoveredTarget = hit;
        _dwellProgress = 0.0;
      } else {
        _dwellProgress += dt / 1.5;
        if (_dwellProgress >= 1.0 || (fistClosed && !_fistCooldown)) {
          _dwellProgress = 0.0;
          _fistCooldown = true;
          Future.delayed(const Duration(milliseconds: 700), () {
            _fistCooldown = false;
          });
          SystemSound.play(SystemSoundType.click);
          hit.onTriggered();
        }
      }
    } else {
      _hoveredTarget = null;
      _dwellProgress = 0.0;
    }
    setState(() {});
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    return Stack(
      children: [
        widget.child,
        if (_hasHand)
          Positioned(
            left: (_smoothedCursor.dx * size.width) - 28,
            top: (_smoothedCursor.dy * size.height) - 28,
            child: IgnorePointer(
              child: SizedBox(
                width: 56,
                height: 56,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    CircularProgressIndicator(
                      value: _dwellProgress > 0 ? _dwellProgress : 1.0,
                      strokeWidth: 5,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        _dwellProgress > 0
                            ? Colors.amberAccent
                            : Colors.cyanAccent,
                      ),
                      backgroundColor: Colors.white24,
                    ),
                    Container(
                      width: 18,
                      height: 18,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.white,
                        boxShadow: [
                          BoxShadow(
                            color: Colors.cyanAccent.withOpacity(0.9),
                            blurRadius: 16,
                            spreadRadius: 4,
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
      ],
    );
  }
}

class DwellButton extends StatefulWidget {
  final String label;
  final IconData icon;
  final Color color;
  final VoidCallback onPressed;
  final double width;
  final double height;

  const DwellButton({
    super.key,
    required this.label,
    required this.icon,
    required this.color,
    required this.onPressed,
    this.width = 240,
    this.height = 84,
  });

  @override
  State<DwellButton> createState() => _DwellButtonState();
}

class _DwellButtonState extends State<DwellButton> with DwellTargetState {
  @override
  VoidCallback get onTriggered => widget.onPressed;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      key: targetKey,
      onTap: widget.onPressed,
      child: Container(
        width: widget.width,
        height: widget.height,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [widget.color, widget.color.withOpacity(0.65)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(22),
          border: Border.all(color: Colors.white70, width: 2.5),
          boxShadow: [
            BoxShadow(
              color: widget.color.withOpacity(0.45),
              blurRadius: 14,
              offset: const Offset(0, 6),
            )
          ],
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(widget.icon, color: Colors.white, size: 32),
            const SizedBox(width: 10),
            Flexible(
              child: Text(
                widget.label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 19,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class SkeletonPainter extends CustomPainter {
  final List<TrackedPlayer> players;
  SkeletonPainter(this.players);

  static const List<List<PoseLandmarkType>> connections = [
    [PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder],
    [PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow],
    [PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist],
    [PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow],
    [PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist],
    [PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip],
    [PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip],
    [PoseLandmarkType.leftHip, PoseLandmarkType.rightHip],
    [PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee],
    [PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle],
    [PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee],
    [PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle],
  ];

  @override
  void paint(Canvas canvas, Size size) {
    for (final player in players) {
      final bonePaint = Paint()
        ..color = player.playerIndex == 0 ? Colors.cyanAccent : Colors.pinkAccent
        ..strokeWidth = 5.0
        ..strokeCap = StrokeCap.round;

      final jointPaint = Paint()..color = Colors.white;

      for (final pair in connections) {
        final p1 = player.getLandmark(pair[0]);
        final p2 = player.getLandmark(pair[1]);
        if (p1 != null &&
            p2 != null &&
            player.isVisible(pair[0], threshold: 0.4) &&
            player.isVisible(pair[1], threshold: 0.4)) {
          canvas.drawLine(
            Offset(p1.dx * size.width, p1.dy * size.height),
            Offset(p2.dx * size.width, p2.dy * size.height),
            bonePaint,
          );
        }
      }

      player.landmarks.forEach((type, pos) {
        if (player.isVisible(type, threshold: 0.4)) {
          canvas.drawCircle(
            Offset(pos.dx * size.width, pos.dy * size.height),
            6.0,
            jointPaint,
          );
        }
      });
    }
  }

  @override
  bool shouldRepaint(covariant SkeletonPainter oldDelegate) => true;
}

// ============================================================================
// SCREEN 1: CALIBRATION & FRAMING GUIDE
// ============================================================================
class CalibrationScreen extends StatefulWidget {
  const CalibrationScreen({super.key});

  @override
  State<CalibrationScreen> createState() => _CalibrationScreenState();
}

class _CalibrationScreenState extends State<CalibrationScreen> {
  StreamSubscription<PoseFrameResult>? _sub;
  List<TrackedPlayer> _players = [];
  FramingStatus _status = FramingStatus.noPlayer;

  @override
  void initState() {
    super.initState();
    _sub = PoseTrackerService.instance.poseStream.listen((frame) {
      if (!mounted) return;
      setState(() {
        _players = frame.players;
        _status = frame.players.isEmpty
            ? FramingStatus.noPlayer
            : frame.players.first.framingStatus;
      });
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  String get _alertBannerText {
    switch (_status) {
      case FramingStatus.noPlayer:
        return "STEP IN FRONT OF THE TV (2–3 METERS BACK)";
      case FramingStatus.headCutOff:
        return "HEAD CUT OFF — STEP BACK OR TILT PHONE UP";
      case FramingStatus.feetCutOff:
        return "FEET NOT VISIBLE — STEP BACK SO FULL BODY FITS";
      case FramingStatus.optimal:
        return "PERFECT FRAMING! HOLD HAND OVER 'ENTER LOBBY' OR CLOSE FIST";
    }
  }

  @override
  Widget build(BuildContext context) {
    final controller = PoseTrackerService.instance.cameraController;
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      body: Row(
        children: [
          Expanded(
            flex: 3,
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(24),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    if (controller != null && controller.value.isInitialized)
                      Transform.scale(
                        scaleX: -1.0,
                        child: CameraPreview(controller),
                      )
                    else
                      const Center(child: CircularProgressIndicator()),
                    CustomPaint(painter: SkeletonPainter(_players)),
                    Container(
                      decoration: BoxDecoration(
                        border: Border.all(
                          color: _status == FramingStatus.optimal
                              ? Colors.greenAccent
                              : Colors.amberAccent,
                          width: 5,
                        ),
                        borderRadius: BorderRadius.circular(24),
                      ),
                    ),
                    Positioned(
                      top: 16,
                      left: 24,
                      right: 24,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            vertical: 12, horizontal: 20),
                        decoration: BoxDecoration(
                          color: (_status == FramingStatus.optimal
                                  ? Colors.green.shade800
                                  : Colors.orange.shade800)
                              .withOpacity(0.92),
                          borderRadius: BorderRadius.circular(14),
                        ),
                        child: Text(
                          _alertBannerText,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          Expanded(
            flex: 2,
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Text(
                    "ROMP MOTION SETUP",
                    style: TextStyle(
                      color: Colors.cyanAccent,
                      fontSize: 28,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    "1. Place phone beneath your TV.\n"
                    "2. Stand 2–3 meters back until head & feet appear inside the box.\n"
                    "3. Raise one hand to move cursor; hold 1.5s or close fist to click.",
                    style: TextStyle(color: Colors.white70, fontSize: 16, height: 1.4),
                  ),
                  const SizedBox(height: 28),
                  DwellButton(
                    label: "ENTER LOBBY",
                    icon: Icons.play_arrow_rounded,
                    color: Colors.green,
                    onPressed: () {
                      Navigator.of(context).pushReplacement(
                        MaterialPageRoute(
                            builder: (_) => const MainMenuScreen()),
                      );
                    },
                  ),
                  const SizedBox(height: 16),
                  DwellButton(
                    label: "TRACKING MEASURE",
                    icon: Icons.speed,
                    color: Colors.deepPurpleAccent,
                    onPressed: () {
                      Navigator.of(context).push(
                        MaterialPageRoute(
                            builder: (_) => const LatencyScreen()),
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// SCREEN 2: TRACKING MEASURE (TV LATENCY CALIBRATION)
// ============================================================================
class LatencyScreen extends StatefulWidget {
  const LatencyScreen({super.key});

  @override
  State<LatencyScreen> createState() => _LatencyScreenState();
}

class _LatencyScreenState extends State<LatencyScreen> {
  StreamSubscription<PoseFrameResult>? _sub;
  bool _isFlashing = false;
  bool _measuring = false;
  int _flashStartMs = 0;
  double _baselineLuma = 0.0;
  final List<double> _lumaSamples = [];
  int _measuredLatencyMs = StorageService.instance.progress.latencyOffsetMs;
  String _statusMessage =
      "Point front camera toward TV reflection and run test.";

  @override
  void initState() {
    super.initState();
    _sub = PoseTrackerService.instance.poseStream.listen(_onCameraFrame);
  }

  void _onCameraFrame(PoseFrameResult frame) {
    if (!_measuring) {
      _lumaSamples.add(frame.averageFrameLuminance);
      if (_lumaSamples.length > 20) _lumaSamples.removeAt(0);
      _baselineLuma = _lumaSamples.isEmpty
          ? 0
          : _lumaSamples.reduce((a, b) => a + b) / _lumaSamples.length;
      return;
    }

    if (_isFlashing && frame.averageFrameLuminance > _baselineLuma + 18.0) {
      final int delta = (frame.timestampMs - _flashStartMs).clamp(30, 450);
      _measuring = false;
      setState(() {
        _isFlashing = false;
        _measuredLatencyMs = delta;
        _statusMessage = "TV + Camera Latency Calibrated: ${delta}ms";
      });
      StorageService.instance.setLatencyOffset(delta);
    }
  }

  Future<void> _startFlashCalibration() async {
    setState(() {
      _statusMessage = "Measuring optical reflection... Hold steady!";
      _isFlashing = false;
      _measuring = false;
    });
    await Future.delayed(const Duration(milliseconds: 600));
    if (!mounted) return;

    setState(() {
      _flashStartMs = DateTime.now().millisecondsSinceEpoch;
      _isFlashing = true;
      _measuring = true;
    });

    Future.delayed(const Duration(milliseconds: 900), () {
      if (mounted && _measuring) {
        setState(() {
          _measuring = false;
          _isFlashing = false;
          _statusMessage =
              "Reflection faint — using safe TV offset (${_measuredLatencyMs}ms).";
        });
      }
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _isFlashing ? Colors.white : const Color(0xFF0D1117),
      body: Center(
        child: _isFlashing
            ? const SizedBox.shrink()
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text(
                    "TRACKING MEASURE (TV LATENCY SYNC)",
                    style: TextStyle(
                      color: Colors.cyanAccent,
                      fontSize: 30,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    _statusMessage,
                    style: const TextStyle(color: Colors.white, fontSize: 20),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    "Saved Offset: ${_measuredLatencyMs} ms",
                    style: const TextStyle(
                      color: Colors.amberAccent,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 36),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      DwellButton(
                        label: "RUN FLASH TEST",
                        icon: Icons.flash_on,
                        color: Colors.amber.shade700,
                        onPressed: _startFlashCalibration,
                      ),
                      const SizedBox(width: 24),
                      DwellButton(
                        label: "BACK",
                        icon: Icons.arrow_back,
                        color: Colors.blueGrey,
                        onPressed: () => Navigator.of(context).pop(),
                      ),
                    ],
                  ),
                ],
              ),
      ),
    );
  }
}

// ============================================================================
// SCREEN 3: MAIN LOBBY & MODE SELECTOR
// ============================================================================
class MainMenuScreen extends StatefulWidget {
  const MainMenuScreen({super.key});

  @override
  State<MainMenuScreen> createState() => _MainMenuScreenState();
}

class _MainMenuScreenState extends State<MainMenuScreen> {
  GameMode _selectedMode = GameMode.solo;

  void _cycleMode() {
    setState(() {
      _selectedMode =
          GameMode.values[(_selectedMode.index + 1) % GameMode.values.length];
    });
  }

  String get _modeLabel {
    switch (_selectedMode) {
      case GameMode.solo:
        return "MODE: SOLO";
      case GameMode.together:
        return "MODE: TOGETHER (2P)";
      case GameMode.takeTurns:
        return "MODE: TAKE TURNS";
    }
  }

  Future<void> _launchGame(Widget screen) async {
    await Navigator.of(context).push(MaterialPageRoute(builder: (_) => screen));
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final progress = StorageService.instance.progress;

    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 18),
          child: Column(
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.bolt, color: Colors.amberAccent, size: 32),
                      const SizedBox(width: 8),
                      Text(
                        "LVL ${progress.level}  •  ${progress.xp} XP",
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.w900,
                        ),
                      ),
                      const SizedBox(width: 24),
                      const Icon(Icons.local_fire_department,
                          color: Colors.deepOrangeAccent, size: 30),
                      const SizedBox(width: 6),
                      Text(
                        "${progress.streakDays} DAY STREAK",
                        style: const TextStyle(
                          color: Colors.orangeAccent,
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  Row(
                    children: [
                      DwellButton(
                        label: _modeLabel,
                        icon: Icons.people_alt_rounded,
                        color: Colors.teal,
                        width: 250,
                        height: 64,
                        onPressed: _cycleMode,
                      ),
                      const SizedBox(width: 14),
                      DwellButton(
                        label: "SETUP",
                        icon: Icons.accessibility_new,
                        color: Colors.blueGrey.shade700,
                        width: 165,
                        height: 64,
                        onPressed: () {
                          Navigator.of(context).pushReplacement(
                            MaterialPageRoute(
                                builder: (_) => const CalibrationScreen()),
                          );
                        },
                      ),
                    ],
                  ),
                ],
              ),
              const Spacer(),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _buildGameCard(
                    title: "SLICE",
                    subtitle: "Arcade Hand Slasher",
                    stars: progress.bestStars['slice'] ?? 0,
                    color: Colors.orange.shade800,
                    icon: Icons.content_cut_rounded,
                    onLaunch: () =>
                        _launchGame(SliceGameScreen(mode: _selectedMode)),
                  ),
                  _buildGameCard(
                    title: "SPRINT",
                    subtitle: "High-Knee 3D Dash",
                    stars: progress.bestStars['sprint'] ?? 0,
                    color: Colors.cyan.shade800,
                    icon: Icons.directions_run_rounded,
                    onLaunch: () =>
                        _launchGame(SprintGameScreen(mode: _selectedMode)),
                  ),
                  _buildGameCard(
                    title: "PULSE",
                    subtitle: "Beat Punch & Squat",
                    stars: progress.bestStars['pulse'] ?? 0,
                    color: Colors.pink.shade700,
                    icon: Icons.music_note_rounded,
                    onLaunch: () =>
                        _launchGame(PulseGameScreen(mode: _selectedMode)),
                  ),
                ],
              ),
              const Spacer(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildGameCard({
    required String title,
    required String subtitle,
    required int stars,
    required Color color,
    required IconData icon,
    required VoidCallback onLaunch,
  }) {
    return Container(
      width: 260,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(26),
        border: Border.all(color: color, width: 3),
      ),
      child: Column(
        children: [
          Icon(icon, size: 54, color: color),
          const SizedBox(height: 10),
          Text(
            title,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 28,
              fontWeight: FontWeight.w900,
            ),
          ),
          Text(
            subtitle,
            style: const TextStyle(color: Colors.white60, fontSize: 14),
          ),
          const SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(
              3,
              (i) => Icon(
                i < stars ? Icons.star_rounded : Icons.star_border_rounded,
                color: Colors.amberAccent,
                size: 28,
              ),
            ),
          ),
          const SizedBox(height: 16),
          DwellButton(
            label: "PLAY",
            icon: Icons.play_arrow_rounded,
            color: color,
            width: 210,
            height: 68,
            onPressed: onLaunch,
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// MINI-GAME 1: SLICE (ARCADE FRUIT & BOMB SLASHER)
// ============================================================================
class _FlyingItem {
  Offset pos;
  Offset vel;
  final bool isBomb;
  final Color color;
  bool sliced = false;

  _FlyingItem({
    required this.pos,
    required this.vel,
    required this.isBomb,
    required this.color,
  });
}

class SliceGameScreen extends StatefulWidget {
  final GameMode mode;
  const SliceGameScreen({super.key, required this.mode});

  @override
  State<SliceGameScreen> createState() => _SliceGameScreenState();
}

class _SliceGameScreenState extends State<SliceGameScreen>
    with SingleTickerProviderStateMixin {
  late Ticker _ticker;
  StreamSubscription<PoseFrameResult>? _poseSub;
  final Random _rng = Random();

  final List<_FlyingItem> _items = [];
  final Map<int, List<Offset>> _trails = {0: [], 1: []};
  final List<int> _scores = [0, 0];

  double _timeLeft = 45.0;
  double _spawnTimer = 0.0;
  Duration _lastElapsed = Duration.zero;
  bool _gameOver = false;
  int _activeTurnPlayer = 0;

  @override
  void initState() {
    super.initState();
    _poseSub = PoseTrackerService.instance.poseStream.listen(_onPoseFrame);
    _ticker = createTicker(_onTick)..start();
  }

  void _onPoseFrame(PoseFrameResult frame) {
    if (_gameOver) return;

    for (final player in frame.players) {
      if (widget.mode == GameMode.takeTurns && player.playerIndex != 0) continue;
      final pIdx = widget.mode == GameMode.takeTurns
          ? _activeTurnPlayer
          : player.playerIndex;

      final rWrist = player.getLandmark(PoseLandmarkType.rightWrist);
      final lWrist = player.getLandmark(PoseLandmarkType.leftWrist);

      for (final wrist in [rWrist, lWrist]) {
        if (wrist == null) continue;
        final trail = _trails[pIdx]!;
        if (trail.isNotEmpty && (wrist - trail.last).distance > 0.025) {
          _checkSliceCollision(wrist, pIdx);
        }
        trail.add(wrist);
        if (trail.length > 10) trail.removeAt(0);
      }
    }
  }

  void _checkSliceCollision(Offset wristPos, int playerIndex) {
    for (final item in _items) {
      if (!item.sliced && (item.pos - wristPos).distance < 0.09) {
        item.sliced = true;
        SystemSound.play(SystemSoundType.click);
        if (item.isBomb) {
          _scores[playerIndex] = max(0, _scores[playerIndex] - 50);
        } else {
          _scores[playerIndex] += 25;
        }
      }
    }
  }

  void _onTick(Duration elapsed) {
    if (_gameOver) return;
    final dt = (_lastElapsed == Duration.zero)
        ? 0.016
        : (elapsed - _lastElapsed).inMicroseconds / 1e6;
    _lastElapsed = elapsed;

    _timeLeft -= dt;
    if (widget.mode == GameMode.takeTurns && _timeLeft < 22.5) {
      _activeTurnPlayer = 1;
    }

    if (_timeLeft <= 0) {
      _finishGame();
      return;
    }

    _spawnTimer -= dt;
    if (_spawnTimer <= 0) {
      _spawnTimer = 0.75;
      _spawnItem();
    }

    for (final item in _items) {
      item.pos += item.vel * dt;
      item.vel += Offset(0, 0.85 * dt);
    }
    _items.removeWhere((i) => i.pos.dy > 1.2 || i.sliced);
    setState(() {});
  }

  void _spawnItem() {
    final bool isBomb = _rng.nextDouble() < 0.22;
    final double startX = 0.2 + _rng.nextDouble() * 0.6;
    final double vx = (0.5 - startX) * (0.5 + _rng.nextDouble() * 0.6);
    final double vy = -0.95 - _rng.nextDouble() * 0.25;
    final colors = [
      Colors.redAccent,
      Colors.orangeAccent,
      Colors.lightGreenAccent,
      Colors.purpleAccent
    ];

    _items.add(_FlyingItem(
      pos: Offset(startX, 1.05),
      vel: Offset(vx, vy),
      isBomb: isBomb,
      color: isBomb ? Colors.grey.shade900 : colors[_rng.nextInt(colors.length)],
    ));
  }

  Future<void> _finishGame() async {
    _gameOver = true;
    _ticker.stop();
    final totalScore = _scores[0] + _scores[1];
    final stars = totalScore >= 400 ? 3 : (totalScore >= 200 ? 2 : 1);
    await StorageService.instance.recordGameResult(
      gameKey: 'slice',
      earnedXp: totalScore ~/ 2 + 50,
      stars: stars,
    );
    setState(() {});
  }

  @override
  void dispose() {
    _ticker.dispose();
    _poseSub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF101522),
      body: Stack(
        children: [
          CustomPaint(
            size: Size.infinite,
            painter: _SliceGamePainter(_items, _trails),
          ),
          Positioned(
            top: 20,
            left: 28,
            right: 28,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  widget.mode == GameMode.solo
                      ? "SCORE: ${_scores[0]}"
                      : "P1: ${_scores[0]}   |   P2: ${_scores[1]}",
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 28,
                      fontWeight: FontWeight.w900),
                ),
                if (widget.mode == GameMode.takeTurns)
                  Text(
                    "TURN: PLAYER ${_activeTurnPlayer + 1}",
                    style: const TextStyle(
                        color: Colors.amberAccent,
                        fontSize: 24,
                        fontWeight: FontWeight.bold),
                  ),
                Text(
                  "${_timeLeft.ceil()}s",
                  style: const TextStyle(
                      color: Colors.cyanAccent,
                      fontSize: 32,
                      fontWeight: FontWeight.w900),
                ),
              ],
            ),
          ),
          if (_gameOver)
            Container(
              color: Colors.black87,
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      "ROUND COMPLETE!",
                      style: TextStyle(
                          color: Colors.amberAccent,
                          fontSize: 40,
                          fontWeight: FontWeight.w900),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      "Final Score: ${_scores[0] + _scores[1]}",
                      style: const TextStyle(color: Colors.white, fontSize: 28),
                    ),
                    const SizedBox(height: 28),
                    DwellButton(
                      label: "BACK TO LOBBY",
                      icon: Icons.home,
                      color: Colors.green,
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _SliceGamePainter extends CustomPainter {
  final List<_FlyingItem> items;
  final Map<int, List<Offset>> trails;
  _SliceGamePainter(this.items, this.trails);

  @override
  void paint(Canvas canvas, Size size) {
    for (final item in items) {
      final center = Offset(item.pos.dx * size.width, item.pos.dy * size.height);
      canvas.drawCircle(
          center, item.isBomb ? 34 : 38, Paint()..color = item.color);

      if (item.isBomb) {
        canvas.drawCircle(
          center,
          34,
          Paint()
            ..color = Colors.red
            ..style = PaintingStyle.stroke
            ..strokeWidth = 4,
        );
      } else {
        canvas.drawCircle(center.translate(-10, -10), 10,
            Paint()..color = Colors.white.withOpacity(0.35));
      }
    }

    trails.forEach((playerIdx, points) {
      if (points.length < 2) return;
      final bladePaint = Paint()
        ..color = playerIdx == 0 ? Colors.cyanAccent : Colors.pinkAccent
        ..strokeWidth = 10
        ..strokeCap = StrokeCap.round;

      for (int i = 0; i < points.length - 1; i++) {
        canvas.drawLine(
          Offset(points[i].dx * size.width, points[i].dy * size.height),
          Offset(points[i + 1].dx * size.width, points[i + 1].dy * size.height),
          bladePaint,
        );
      }
    });
  }

  @override
  bool shouldRepaint(covariant _SliceGamePainter oldDelegate) => true;
}

// ============================================================================
// MINI-GAME 2: SPRINT (HIGH-KNEE 3D TRACK RUNNER)
// ============================================================================
class SprintGameScreen extends StatefulWidget {
  final GameMode mode;
  const SprintGameScreen({super.key, required this.mode});

  @override
  State<SprintGameScreen> createState() => _SprintGameScreenState();
}

class _SprintGameScreenState extends State<SprintGameScreen>
    with SingleTickerProviderStateMixin {
  late Ticker _ticker;
  StreamSubscription<PoseFrameResult>? _poseSub;

  final List<double> _distances = [0.0, 0.0];
  final List<double> _speeds = [0.0, 0.0];
  final List<bool> _lastLeftKneeUp = [false, false];
  double _trackScroll = 0.0;
  double _elapsedSeconds = 0.0;
  Duration _lastTick = Duration.zero;
  bool _finished = false;

  @override
  void initState() {
    super.initState();
    _poseSub = PoseTrackerService.instance.poseStream.listen(_onPoseFrame);
    _ticker = createTicker(_onTick)..start();
  }

  void _onPoseFrame(PoseFrameResult frame) {
    if (_finished) return;

    for (final player in frame.players) {
      final pIdx = player.playerIndex;
      final lHip = player.getLandmark(PoseLandmarkType.leftHip);
      final rHip = player.getLandmark(PoseLandmarkType.rightHip);
      final lKnee = player.getLandmark(PoseLandmarkType.leftKnee);
      final rKnee = player.getLandmark(PoseLandmarkType.rightKnee);

      if (lHip == null || rHip == null || lKnee == null || rKnee == null) {
        continue;
      }

      final double hipY = (lHip.dy + rHip.dy) / 2.0;
      final bool leftHigh = (hipY - lKnee.dy) > -0.06;
      final bool rightHigh = (hipY - rKnee.dy) > -0.06;

      if (leftHigh && !_lastLeftKneeUp[pIdx]) {
        _lastLeftKneeUp[pIdx] = true;
        _speeds[pIdx] = (_speeds[pIdx] + 3.2).clamp(0.0, 12.0);
      } else if (rightHigh && _lastLeftKneeUp[pIdx]) {
        _lastLeftKneeUp[pIdx] = false;
        _speeds[pIdx] = (_speeds[pIdx] + 3.2).clamp(0.0, 12.0);
      }
    }
  }

  void _onTick(Duration elapsed) {
    if (_finished) return;
    final dt = (_lastTick == Duration.zero)
        ? 0.016
        : (elapsed - _lastTick).inMicroseconds / 1e6;
    _lastTick = elapsed;

    _elapsedSeconds += dt;
    for (int i = 0; i < 2; i++) {
      _distances[i] += _speeds[i] * dt;
      _speeds[i] = (_speeds[i] - 4.5 * dt).clamp(0.0, 12.0);
    }
    _trackScroll += (_speeds[0] + _speeds[1]) * dt * 0.25;

    final bool p1Done = _distances[0] >= 100.0;
    final bool p2Done =
        widget.mode == GameMode.together && _distances[1] >= 100.0;

    if (p1Done || p2Done) {
      _completeSprint();
      return;
    }
    setState(() {});
  }

  Future<void> _completeSprint() async {
    _finished = true;
    _ticker.stop();
    final stars =
        _elapsedSeconds <= 18.0 ? 3 : (_elapsedSeconds <= 26.0 ? 2 : 1);
    await StorageService.instance.recordGameResult(
      gameKey: 'sprint',
      earnedXp: stars * 100,
      stars: stars,
    );
    setState(() {});
  }

  @override
  void dispose() {
    _ticker.dispose();
    _poseSub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A192F),
      body: Stack(
        children: [
          CustomPaint(
            size: Size.infinite,
            painter: _SprintTrackPainter(
              scrollOffset: _trackScroll,
              twoPlayer: widget.mode == GameMode.together,
            ),
          ),
          Positioned(
            top: 20,
            left: 28,
            right: 28,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  "TIME: ${_elapsedSeconds.toStringAsFixed(1)}s",
                  style: const TextStyle(
                      color: Colors.amberAccent,
                      fontSize: 30,
                      fontWeight: FontWeight.w900),
                ),
                Text(
                  "DISTANCE: ${_distances[0].toStringAsFixed(0)}m / 100m",
                  style: const TextStyle(
                      color: Colors.cyanAccent,
                      fontSize: 28,
                      fontWeight: FontWeight.bold),
                ),
              ],
            ),
          ),
          if (_finished)
            Container(
              color: Colors.black87,
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      "FINISH LINE!",
                      style: TextStyle(
                          color: Colors.cyanAccent,
                          fontSize: 42,
                          fontWeight: FontWeight.w900),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      "100m Time: ${_elapsedSeconds.toStringAsFixed(2)} seconds",
                      style: const TextStyle(color: Colors.white, fontSize: 26),
                    ),
                    const SizedBox(height: 28),
                    DwellButton(
                      label: "BACK TO LOBBY",
                      icon: Icons.home,
                      color: Colors.green,
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _SprintTrackPainter extends CustomPainter {
  final double scrollOffset;
  final bool twoPlayer;

  _SprintTrackPainter({required this.scrollOffset, required this.twoPlayer});

  @override
  void paint(Canvas canvas, Size size) {
    final horizonY = size.height * 0.35;
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, horizonY),
        Paint()..color = const Color(0xFF112240));

    final trackPath = Path()
      ..moveTo(size.width * 0.38, horizonY)
      ..lineTo(size.width * 0.62, horizonY)
      ..lineTo(size.width * 0.92, size.height)
      ..lineTo(size.width * 0.08, size.height)
      ..close();

    canvas.drawPath(trackPath, Paint()..color = const Color(0xFFD9534F));

    final stripePaint = Paint()
      ..color = Colors.white54
      ..strokeWidth = 4;

    for (int i = 0; i < 8; i++) {
      final double t = ((i / 8.0) + (scrollOffset % 1.0)) % 1.0;
      final double persp = t * t;
      final double y = horizonY + (size.height - horizonY) * persp;
      final double halfW = size.width * (0.12 + 0.30 * persp);
      canvas.drawLine(Offset(size.width * 0.5 - halfW, y),
          Offset(size.width * 0.5 + halfW, y), stripePaint);
    }

    canvas.drawCircle(
      Offset(twoPlayer ? size.width * 0.42 : size.width * 0.5, size.height * 0.78),
      32,
      Paint()..color = Colors.cyanAccent,
    );
    if (twoPlayer) {
      canvas.drawCircle(Offset(size.width * 0.58, size.height * 0.78), 32,
          Paint()..color = Colors.pinkAccent);
    }
  }

  @override
  bool shouldRepaint(covariant _SprintTrackPainter oldDelegate) => true;
}

// ============================================================================
// MINI-GAME 3: PULSE (RHYTHM PUNCH & SQUAT TRAINER)
// ============================================================================
enum _PulseTargetType { leftPunch, rightPunch, squatBar }

class _BeatTarget {
  final double hitTimeSec;
  final _PulseTargetType type;
  bool resolved = false;
  _BeatTarget({required this.hitTimeSec, required this.type});
}

class PulseGameScreen extends StatefulWidget {
  final GameMode mode;
  const PulseGameScreen({super.key, required this.mode});

  @override
  State<PulseGameScreen> createState() => _PulseGameScreenState();
}

class _PulseGameScreenState extends State<PulseGameScreen>
    with SingleTickerProviderStateMixin {
  late Ticker _ticker;
  StreamSubscription<PoseFrameResult>? _poseSub;
  final List<_BeatTarget> _targets = [];

  double _songTimeSec = 0.0;
  double _lastBeatIndex = -1;
  Duration _lastTick = Duration.zero;
  int _score = 0;
  int _combo = 0;
  bool _finished = false;

  @override
  void initState() {
    super.initState();
    final rng = Random(42);
    for (double t = 2.0; t <= 40.0; t += 1.0) {
      final types = _PulseTargetType.values;
      _targets.add(
          _BeatTarget(hitTimeSec: t, type: types[rng.nextInt(types.length)]));
    }
    _poseSub = PoseTrackerService.instance.poseStream.listen(_onPoseFrame);
    _ticker = createTicker(_onTick)..start();
  }

  void _onPoseFrame(PoseFrameResult frame) {
    if (_finished || frame.players.isEmpty) return;

    final double latencySec =
        StorageService.instance.progress.latencyOffsetMs / 1000.0;
    final double compensatedTime = _songTimeSec - latencySec;

    for (final player in frame.players) {
      final lWrist = player.getLandmark(PoseLandmarkType.leftWrist);
      final rWrist = player.getLandmark(PoseLandmarkType.rightWrist);
      final nose = player.getLandmark(PoseLandmarkType.nose);

      for (final target in _targets) {
        if (target.resolved) continue;
        if ((target.hitTimeSec - compensatedTime).abs() > 0.25) continue;

        bool hit = false;
        if (target.type == _PulseTargetType.leftPunch &&
            lWrist != null &&
            (lWrist - const Offset(0.28, 0.50)).distance < 0.16) {
          hit = true;
        } else if (target.type == _PulseTargetType.rightPunch &&
            rWrist != null &&
            (rWrist - const Offset(0.72, 0.50)).distance < 0.16) {
          hit = true;
        } else if (target.type == _PulseTargetType.squatBar &&
            nose != null &&
            nose.dy > 0.56) {
          hit = true;
        }

        if (hit) {
          target.resolved = true;
          _combo++;
          _score += 50 + (_combo * 5);
          SystemSound.play(SystemSoundType.click);
        }
      }
    }
  }

  void _onTick(Duration elapsed) {
    if (_finished) return;
    final dt = (_lastTick == Duration.zero)
        ? 0.016
        : (elapsed - _lastTick).inMicroseconds / 1e6;
    _lastTick = elapsed;

    _songTimeSec += dt;
    final double currentBeat = (_songTimeSec * 2).floorToDouble();
    if (currentBeat > _lastBeatIndex) {
      _lastBeatIndex = currentBeat;
      SystemSound.play(SystemSoundType.click); // Built-in 120 BPM metronome pulse
    }

    for (final t in _targets) {
      if (!t.resolved && _songTimeSec - t.hitTimeSec > 0.30) {
        t.resolved = true;
        _combo = 0;
      }
    }

    if (_songTimeSec >= 42.0) {
      _finishPulse();
      return;
    }
    setState(() {});
  }

  Future<void> _finishPulse() async {
    _finished = true;
    _ticker.stop();
    final stars = _score >= 1500 ? 3 : (_score >= 800 ? 2 : 1);
    await StorageService.instance.recordGameResult(
      gameKey: 'pulse',
      earnedXp: _score ~/ 5 + 60,
      stars: stars,
    );
    setState(() {});
  }

  @override
  void dispose() {
    _ticker.dispose();
    _poseSub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF140B24),
      body: Stack(
        children: [
          CustomPaint(
            size: Size.infinite,
            painter: _PulseGamePainter(_targets, _songTimeSec),
          ),
          Positioned(
            top: 20,
            left: 28,
            right: 28,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  "SCORE: $_score",
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 28,
                      fontWeight: FontWeight.w900),
                ),
                Text(
                  "COMBO: ${_combo}x",
                  style: const TextStyle(
                      color: Colors.pinkAccent,
                      fontSize: 28,
                      fontWeight: FontWeight.w900),
                ),
              ],
            ),
          ),
          if (_finished)
            Container(
              color: Colors.black87,
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      "BEAT SESSION COMPLETE!",
                      style: TextStyle(
                          color: Colors.pinkAccent,
                          fontSize: 38,
                          fontWeight: FontWeight.w900),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      "Final Rhythm Score: $_score",
                      style: const TextStyle(color: Colors.white, fontSize: 26),
                    ),
                    const SizedBox(height: 28),
                    DwellButton(
                      label: "BACK TO LOBBY",
                      icon: Icons.home,
                      color: Colors.green,
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _PulseGamePainter extends CustomPainter {
  final List<_BeatTarget> targets;
  final double songTimeSec;
  _PulseGamePainter(this.targets, this.songTimeSec);

  @override
  void paint(Canvas canvas, Size size) {
    final leftRing = Offset(size.width * 0.28, size.height * 0.50);
    final rightRing = Offset(size.width * 0.72, size.height * 0.50);

    canvas.drawCircle(
      leftRing,
      54,
      Paint()
        ..color = Colors.cyanAccent
        ..style = PaintingStyle.stroke
        ..strokeWidth = 6,
    );
    canvas.drawCircle(
      rightRing,
      54,
      Paint()
        ..color = Colors.pinkAccent
        ..style = PaintingStyle.stroke
        ..strokeWidth = 6,
    );

    for (final t in targets) {
      if (t.resolved) continue;
      final double timeUntilHit = t.hitTimeSec - songTimeSec;
      if (timeUntilHit < -0.2 || timeUntilHit > 1.8) continue;

      final double progress = (1.0 - (timeUntilHit / 1.8)).clamp(0.0, 1.2);
      final double radius = 14 + (40 * progress);

      if (t.type == _PulseTargetType.leftPunch) {
        final pos = Offset.lerp(
            Offset(size.width * 0.5, size.height * 0.35), leftRing, progress)!;
        canvas.drawCircle(
            pos, radius, Paint()..color = Colors.cyanAccent.withOpacity(0.85));
      } else if (t.type == _PulseTargetType.rightPunch) {
        final pos = Offset.lerp(
            Offset(size.width * 0.5, size.height * 0.35), rightRing, progress)!;
        canvas.drawCircle(
            pos, radius, Paint()..color = Colors.pinkAccent.withOpacity(0.85));
      } else {
        final double y = size.height * 0.32 * progress;
        canvas.drawRect(
          Rect.fromLTWH(size.width * 0.20, y, size.width * 0.60, 26),
          Paint()..color = Colors.amberAccent.withOpacity(0.9),
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant _PulseGamePainter oldDelegate) => true;
}
