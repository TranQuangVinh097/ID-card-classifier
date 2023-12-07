package com.example.face_detector;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;

import com.example.face_detector.camera.CameraManager;
import com.example.face_detector.camera.GraphicOverlay;

public class MainActivity extends AppCompatActivity {

    private CameraManager cameraManager;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        PreviewView cameraPreview = findViewById(R.id.previewView_finder);
        GraphicOverlay cameraOverlay = findViewById(R.id.graphicOverlay_finder);
        Button switchCamera = findViewById(R.id.btn_switch_camera);

        TextView tv = findViewById(R.id.aps);
        checkCameraPermission();
        cameraManager = new CameraManager(this, this, cameraPreview, cameraOverlay, tv);
        switchCamera.setOnClickListener(v -> cameraManager.setCameraLen());
    }


    private void checkCameraPermission() {
        if (ActivityCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 0);
    }
}