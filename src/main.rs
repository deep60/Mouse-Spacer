use enigo::{Enigo, KeyboardControllable, MouseControllable};
use nokhwa::{Camera, CameraFormat, FrameFront}; //For Webcam capture
use tch::{nn, Device, Tensor};

#[derive(Debug)]
struct HandGestureController {
    camera: Camera,
    hand_landmarker: HandLandmarker,
    model: nn::Module,
    enigo: Enigo,
    status: i32,
    last_distances: f32,
    last_coord_x: i32,
    last_coord_y: i32,
    is_clicked: bool,
}

impl HandGestureController {
    fn new() -> Result<Self, Box<dyn Error>> {
        //Initialize Camera
        let camera = Camera::new(0, CameraFormat::new_from(640, 480, FrameFront::MJPEG, 30))?;

        // Initialize Mediapipe hand landmarker
        let hand_landmarker = HandLandmarker::new(RunningMode::LiveStream, 1, 0.7)?;
        // Load the model
        let model = tch::CModule::load("./my_model/model.pt")?;

        Ok(HandGestureController {
            camera,
            hand_landmarker,
            model,
            enigo: Enigo::new(),
            status: -1,
            last_distances: -1.0,
            last_coord_x: 0,
            last_coord_y: 0,
            is_clicked: false,
        })
    }

    fn calculate_distances(&self, point1: Point, point2: Point) -> f32 {
        let dx = point1.x - point2.x;
        let dy = point1.y - point2.y;
        ((dx * dx + dy * dy) as f32).sqrt()
    }

    fn process_frame(&mut self) -> Result<bool, Box<dyn Error>> {
        let frame = self.camera.frame()?;
        let mut image = Mat::from_slices(&frame)?;
        flip(&image, &mut image, 1)?;

        //process hand landmarks
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut controller = HandGestureController::new()?;
    controller.run()?;
    Ok(())
}
