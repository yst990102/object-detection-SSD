import gradio as gr
import numpy as np
import imutils
import cv2
import uuid
import atexit
import shutil
import os

# 定义视频处理函数
def process_video(input_video_path):
    use_gpu = False

    frame_display_interval = 30
    frame_display_counter = frame_display_interval
    confidence_level = 0.5
    writer = None

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt', 'ssd_files/MobileNetSSD_deploy.caffemodel')

    if use_gpu:
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(input_video_path)

    while True:
        ret, frame = vs.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_level:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)

        # 将帧调整回与输入视频相同的分辨率
        frame = imutils.resize(frame, width=w, height=h)

        # 如果没有创建视频编写器，请创建它
        if writer is None:
            random_filename = "./tmp/" + str(uuid.uuid4()) + ".mp4"
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 可以根据需要选择适当的编解码器
            writer = cv2.VideoWriter(random_filename, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # 将帧写入输出视频
        writer.write(frame)

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB
        if frame_display_counter:
            frame_display_counter -= 1
        else:
            yield display_frame, None
            frame_display_counter = frame_display_interval

    vs.release()
    writer.release()
    yield display_frame, random_filename

# 定义一个清理函数，用于删除tmp文件夹及其内容
def cleanup_tmp_folder():
    try:
        shutil.rmtree("./tmp")
    except Exception as e:
        print(f"Error cleaning up tmp folder: {str(e)}")

def main():
    # 定义Gradio界面
    with gr.Blocks() as demo:
        with gr.Row():
            input_video = gr.Video(label="input")
            processed_frames = gr.Image(label="live preview")
            output_video = gr.Video(label="output")

        with gr.Row():
            examples = gr.Examples(["test/test.mp4"], inputs=input_video)
            process_video_btn = gr.Button("process video")

        process_video_btn.click(process_video, input_video, [processed_frames, output_video])

    demo.queue().launch(share=True,debug=True)

if __name__ == "__main__":
    # 注册清理函数，确保在程序退出时执行
    atexit.register(cleanup_tmp_folder)
    
    # 根目录路径
    root_directory = "./"
    # tmp文件夹路径
    tmp_directory = os.path.join(root_directory, "tmp")
    # 检查tmp文件夹是否存在，如果不存在则创建它
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    # 执行main()
    main()