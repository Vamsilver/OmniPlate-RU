@echo off
title Export YOLO-Pose to ONNX
cd /d D:\AIProjects\VolgaIT
echo ========================================================
echo Exporting detector_yolo_pose_best.pt to ONNX...
echo ========================================================
.venv\Scripts\python.exe -c "from ultralytics import YOLO; import shutil, os; m = YOLO('models/detector_yolo_pose_best.pt'); out = m.export(format='onnx', imgsz=640, simplify=True); print('Exported to:', out); shutil.copy(out, 'models/detector_yolo_pose.onnx') if os.path.exists(out) else None; print('[SUCCESS] models/detector_yolo_pose.onnx ready!')"
pause
