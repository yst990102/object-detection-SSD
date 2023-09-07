import gradio as gr
import numpy as np
import imutils
import cv2
import uuid
import atexit
import shutil
import os
import skvideo.io

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
        origin_frame = frame.copy()
        origin_h, origin_w = origin_frame.shape[:2]
        frame = imutils.resize(frame, width=400, height=400)
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
                originStartX = int(startX*origin_w/w)
                originStartY = int(startY*origin_h/h)
                originEndX = int(endX*origin_w/w)
                originEndY = int(endY*origin_h/h)

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(origin_frame, (originStartX, originStartY), (originEndX, originEndY), COLORS[idx], 2)

                y = originStartY - 15 if originStartY - 15 > 15 else originStartY + 15
                cv2.putText(origin_frame, label, (originStartX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)

        # 如果没有创建视频编写器，请创建它
        if writer is None:
            random_filename = "./tmp/" + str(uuid.uuid4()) + ".mp4"
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 可以根据需要选择适当的编解码器
            # writer = cv2.VideoWriter(random_filename, fourcc, 30, (origin_frame.shape[1], origin_frame.shape[0]), True)
            writer = skvideo.io.FFmpegWriter(random_filename, outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p'})
        # 将帧写入输出视频
        # writer.write(origin_frame)
        writer.writeFrame(cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB))

        display_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB
        if frame_display_counter:
            frame_display_counter -= 1
        else:
            yield display_frame, None
            frame_display_counter = frame_display_interval

    vs.release()
    # 在循环结束后释放视频编写器
    # if writer is not None:
    #     writer.release()
    writer.close()
    
    # 将相对路径转换为绝对路径
    absolute_path = os.path.abspath(random_filename)
    print(absolute_path)
    
    # 指定根目录路径
    root_directory = "./tmp"
    # 调用函数打印根目录的文件树
    print_directory_tree(root_directory)

    yield display_frame, absolute_path


def print_directory_tree(directory):
    for root, dirs, files in os.walk(directory):
        # 打印当前目录路径
        print(f"Directory: {root}")
        
        # 打印当前目录下的所有文件
        for file in files:
            print(f"  File: {os.path.join(root, file)}")
        
        # 打印当前目录下的所有子目录
        for dir in dirs:
            print(f"  Subdirectory: {os.path.join(root, dir)}")


# 定义一个清理函数，用于删除tmp文件夹中的.mp4输出视频
def cleanup_tmp_folder():
    tmp_directory = "./tmp"
    try:
        for filename in os.listdir(tmp_directory):
            if filename.endswith(".mp4"):
                file_path = os.path.join(tmp_directory, filename)
                os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up .mp4 files in tmp folder: {str(e)}")

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

    demo.queue().launch(share=False,debug=True,server_name="0.0.0.0",server_port=7860)

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