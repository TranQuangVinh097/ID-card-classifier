package it.polocorese.aicamera;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.core.content.FileProvider;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.label.ImageLabel;
import com.google.mlkit.vision.label.ImageLabeler;
import com.google.mlkit.vision.label.ImageLabeling;
import com.google.mlkit.common.model.LocalModel;
import com.google.mlkit.vision.label.custom.CustomImageLabelerOptions;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.Date;


public class StorageActivity extends AppCompatActivity {

    private File photoFile;
    private static final String TAG = StorageActivity.class.getName();
    TextView Label;
    final int TAKE_PHOTO = 1;
    final int FROM_STORAGE = 2;

    private CustomImageLabelerOptions customImageLabelerOptions;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        File directory = new File(getFilesDir(), "photos");
        if (!directory.exists()) {
            directory.mkdir();
        }

        LocalModel localModel =
                new LocalModel.Builder()
                        .setAssetFilePath("mobilenetv2_050.tflite")
                        .build();

        customImageLabelerOptions =
                new CustomImageLabelerOptions.Builder(localModel)
                        .setConfidenceThreshold(0.0f)
                        .setMaxResultCount(6)
                        .build();


        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.setDisplayHomeAsUpEnabled(true);
        }

        photoFile = new File(directory, "photo.jpg");
        setContentView(R.layout.activity_storage);
        Label = findViewById(R.id.Label);
        Button pickPicture = findViewById(R.id.btn_pick_pic);
        pickPicture.setOnClickListener(v -> onPickImage());

    }

    private  void onPickImage() {
        Uri uri = FileProvider.getUriForFile(this, "it.polocorese.aicamera.fileprovider", photoFile);
        Intent intent = new Intent();

        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        intent = Intent.createChooser(intent,"Select files");

        intent.putExtra(MediaStore.EXTRA_OUTPUT, uri);
        if (intent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(intent, FROM_STORAGE);
        }
    }

    public static double calculateInSampleSize(
            BitmapFactory.Options options, int TARGET_WIDTH, int TARGET_HEIGHT) {
        double inSampleSize = 0;
        boolean scaleByHeight = Math.abs(options.outHeight - TARGET_HEIGHT) >= Math.abs(options.outWidth - TARGET_WIDTH);


        inSampleSize = scaleByHeight
                ? options.outHeight / TARGET_HEIGHT
                : options.outWidth / TARGET_WIDTH;

        return inSampleSize;
    }


    @Override
    protected void onActivityResult(int RequestCode, int resultCode, Intent data) {

        super.onActivityResult(RequestCode, resultCode, data);
        {
            if (resultCode == Activity.RESULT_OK) {
                //Bitmap bitmap = (Bitmap) data.getExtras().get("data");
                Bitmap bitmap = null;
                Bitmap InputMap = null;
                int TARGET_HEIGHT = 800 * 2;
                int TARGET_WIDTH = 405 * 2;
                Matrix matrix = new Matrix();
                int Rotation = 0;

                if (RequestCode == TAKE_PHOTO) {
                    InputMap = BitmapFactory.decodeFile(photoFile.getAbsolutePath());
                    BitmapFactory.Options options = new BitmapFactory.Options();
                    options.inJustDecodeBounds = true;
                    BitmapFactory.decodeFile(photoFile.getAbsolutePath(), options);
                    double sampleSize = 0;
                    if (options.outWidth > options.outHeight) {
                        sampleSize = calculateInSampleSize(options, TARGET_HEIGHT, TARGET_WIDTH);
                        Rotation = 90;
                    }
                    else {
                        //Rotation = 90;
                        sampleSize = calculateInSampleSize(options, TARGET_WIDTH, TARGET_HEIGHT);
                    }

                    options.inSampleSize = (int)Math.pow(2d,
                            Math.floor(Math.log(sampleSize)/Math.log(2d)));

                    options.inJustDecodeBounds = false;
                    bitmap = BitmapFactory.decodeFile(photoFile.getAbsolutePath(), options);
                    matrix.postRotate(Rotation);
                    bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
                } else {
                    Uri selectedImageUri = data.getData();

                    if (selectedImageUri != null) {
                        try {

                            InputStream inputStream = getContentResolver().openInputStream(selectedImageUri);

                            inputStream = getContentResolver().openInputStream(selectedImageUri);
                            BitmapFactory.Options options = new BitmapFactory.Options();
                            BitmapFactory.Options optionsMap = new BitmapFactory.Options();
                            options.inJustDecodeBounds = true;
                            optionsMap.inJustDecodeBounds = true;

                            BitmapFactory.decodeStream(inputStream, null, options);
                            BitmapFactory.decodeStream(inputStream, null, optionsMap);
                            Boolean scaleByHeight = Math.abs(options.outHeight - TARGET_HEIGHT) >= Math.abs(options.outWidth - TARGET_WIDTH);
                            double sampleSize = 0;
                            double MapSampleSize = calculateInSampleSize(optionsMap,288,288);
                            if (options.outWidth > options.outHeight) {

                                sampleSize = scaleByHeight
                                        ? options.outHeight / TARGET_WIDTH
                                        : options.outWidth / TARGET_HEIGHT;
                                Rotation = 90;
                            }
                            else {
                                sampleSize = scaleByHeight
                                        ? options.outHeight / TARGET_HEIGHT
                                        : options.outWidth / TARGET_WIDTH;


                            }

                            options.inSampleSize = (int)Math.pow(2d,
                                    Math.floor(Math.log(sampleSize)/Math.log(2d)));
                            optionsMap.inSampleSize = (int)Math.pow(2d,
                                    Math.floor(Math.log(MapSampleSize)/Math.log(2d)));
                            options.inJustDecodeBounds = false;
                            optionsMap.inJustDecodeBounds = false;
                            inputStream = getContentResolver().openInputStream(selectedImageUri);

                            bitmap = BitmapFactory.decodeStream(inputStream, null, options);
                            inputStream = getContentResolver().openInputStream(selectedImageUri);
                            InputMap = BitmapFactory.decodeStream(inputStream, null, optionsMap);
                            matrix.postRotate(Rotation);
                            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

                        } catch (FileNotFoundException e) {
                            throw new RuntimeException(e);
                        }
                    }
                    else {
                        Toast.makeText(getApplicationContext(), "pain", Toast.LENGTH_LONG).show();
                    }
                }
                if (bitmap != null) {
                    Log.i("MYTAG", "IMAGE RESOLUTION: " + bitmap.getWidth() + " x " + bitmap.getHeight());

                    ImageView ivPhoto = findViewById(R.id.iv_photo);
                    //ivPhoto.setRotation(ivPhoto.getRotation() + Rotation);
                    ivPhoto.setImageBitmap(bitmap);




                    ImageLabeler labeler = ImageLabeling.getClient(customImageLabelerOptions);
                    InputImage inputImage = InputImage.fromBitmap(InputMap, 0);
                    labeler.process(inputImage).addOnSuccessListener(imageLabels -> {
                        String message = "Label:\n";
                        message += " - ";
                        int lbl = 1;
                        for (ImageLabel label : imageLabels) {
                            String getLabel = label.getText();
                            String test =  "None";
                            message += getLabel + " (" + (Math.round(label.getConfidence() * 100000) / 1000) + "%)" + " ";
//                            if (getLabel.compareTo(test) != 0) {
//                                lbl = 0;
//                                message += getLabel + " (" + label.getConfidence() + "%)" + " ";
//                            }
                        }
                        if (lbl == 0)
                            message = "Image is not related";
                        Label.setText(message);
                    });

                    labeler.process(inputImage).addOnFailureListener(imageLabels -> {
                        String message = "Label:\n";
                        message += " " + imageLabels;

                        Label.setText(message);
                    });
                    // Cleanup
                    photoFile.delete();
                }
            }
        }
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                return true;
        }
        return super.onOptionsItemSelected(item);
    }
    public boolean onCreateOptionsMenu(Menu menu) {
        return true;
    }

}