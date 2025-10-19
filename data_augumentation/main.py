import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import random
import io
from datetime import datetime
import zipfile
from io import BytesIO

# 设置页面配置
st.set_page_config(
    page_title="视频/图像数据增强工具",
    layout="wide"
)

# 初始化session_state
if 'extracted_frames' not in st.session_state:
    st.session_state.extracted_frames = []
if 'enhanced_results' not in st.session_state:
    st.session_state.enhanced_results = []
if 'enhanced_names' not in st.session_state:
    st.session_state.enhanced_names = []
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None

# 标题和说明
st.title("视频/图像数据增强工具")
st.write("上传视频或图像文件，进行取帧（视频）和多种图像增强处理")


# 定义图像增强函数
def adjust_brightness(image, factor=1.5):
    """调整亮度"""
    img = image.copy().astype(np.float32)
    img = img * factor
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def decrease_brightness(image, factor=0.5):
    """降低亮度"""
    return adjust_brightness(image, factor)


def gaussian_blur(image, ksize=(5, 5)):
    """高斯模糊"""
    return cv2.GaussianBlur(image, ksize, 0)


def add_noise(image, intensity=15):
    """添加高斯噪声"""
    img = image.copy()
    mean = 0
    sigma = intensity
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = img + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_occlusion(image, occlude_size=(50, 50)):
    """添加随机遮挡，可自定义方块大小"""
    img = image.copy()
    h, w = img.shape[:2]

    # 确保遮挡尺寸不超过图像尺寸
    block_width = min(occlude_size[0], w)
    block_height = min(occlude_size[1], h)

    # 随机位置（确保遮挡块完全在图像内）
    x = random.randint(0, w - block_width)
    y = random.randint(0, h - block_height)

    # 随机遮挡颜色（黑色或白色）
    color = [0, 0, 0] if random.random() < 0.5 else [255, 255, 255]

    img[y:y + block_height, x:x + block_width] = color
    return img


def rotate_image(image, angle=45):
    """旋转图像"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def flip_image(image, direction='horizontal'):
    """翻转图像"""
    if direction == 'horizontal':
        return cv2.flip(image, 1)
    else:  # vertical
        return cv2.flip(image, 0)


# 视频取帧函数
def extract_frames(video_path, num_frames=5):
    """从视频中提取指定数量的帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        st.error("无法读取视频文件，请检查文件是否有效")
        return []

    # 计算需要提取的帧的位置
    frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 转换为RGB格式（OpenCV默认BGR）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


# 处理图像增强的函数
def apply_enhancements(image, selected_methods, occlusion_size=(50, 50)):
    """应用选中的图像增强方法"""
    enhanced_images = []
    method_names = []

    for method in selected_methods:
        if method == "亮度增强":
            enhanced = adjust_brightness(image)
            method_names.append("亮度增强")
        elif method == "亮度降低":
            enhanced = decrease_brightness(image)
            method_names.append("亮度降低")
        elif method == "高斯模糊":
            enhanced = gaussian_blur(image)
            method_names.append("高斯模糊")
        elif method == "高斯噪声":
            enhanced = add_noise(image)
            method_names.append("高斯噪声")
        elif method == "模拟遮挡":
            enhanced = add_occlusion(image, occlude_size=occlusion_size)
            method_names.append("模拟遮挡")
        elif method == "旋转（逆时针45度）":
            enhanced = rotate_image(image)
            method_names.append("旋转（逆时针45度）")
        elif method == "水平翻转":
            enhanced = flip_image(image, 'horizontal')
            method_names.append("水平翻转")
        elif method == "垂直翻转":
            enhanced = flip_image(image, 'vertical')
            method_names.append("垂直翻转")

        enhanced_images.append(enhanced)

    return enhanced_images, method_names


# 图像转字节函数以便下载
def image_to_bytes(image, format='PNG'):
    """将图像转换为字节流以便下载"""
    buf = io.BytesIO()
    img = Image.fromarray(image)
    img.save(buf, format=format)
    return buf.getvalue()


# 创建ZIP文件的函数
def create_zip(images, names=None, format='PNG'):
    """将多个图像打包成ZIP文件"""
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, img in enumerate(images):
            img_bytes = image_to_bytes(img, format)
            if names and i < len(names):
                filename = f"{names[i]}.{format.lower()}"
            else:
                filename = f"image_{i + 1}.{format.lower()}"
            zipf.writestr(filename, img_bytes)

    zip_buffer.seek(0)
    return zip_buffer


# 主程序
def main():
    # 上传文件
    uploaded_file = st.file_uploader("选择图像或视频文件",
                                     type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # 获取文件扩展名
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        # 处理视频文件
        if file_ext in [".mp4", ".avi", ".mov", ".mkv"]:
            st.subheader("视频处理")

            # 保存上传的视频到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # 视频取帧参数设置
            col1, col2 = st.columns(2)
            with col1:
                num_frames = st.slider("选择要提取的帧数", 1, 20, 5)

            with col2:
                extract_btn = st.button("提取帧")

            # 提取帧并保存到session_state
            if extract_btn:
                with st.spinner("正在提取视频帧..."):
                    st.session_state.extracted_frames = extract_frames(tmp_path, num_frames)

                if st.session_state.extracted_frames:
                    st.success(f"成功提取 {len(st.session_state.extracted_frames)} 帧")

            # 删除临时文件
            try:
                os.unlink(tmp_path)
            except:
                pass

            # 显示提取的帧
            if hasattr(st.session_state, 'extracted_frames') and st.session_state.extracted_frames:
                st.subheader("提取的帧预览")
                cols = st.columns(3)
                for i, frame in enumerate(st.session_state.extracted_frames):
                    with cols[i % 3]:
                        st.image(frame, caption=f"帧 {i + 1}", use_column_width=True)

                # 提供ZIP下载
                zip_buffer = create_zip(st.session_state.extracted_frames,
                                        [f"frame_{i + 1}" for i in range(len(st.session_state.extracted_frames))])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="下载所有帧 (ZIP)",
                    data=zip_buffer,
                    file_name=f"extracted_frames_{timestamp}.zip",
                    mime="application/zip"
                )

        # 处理图像文件
        else:
            # 直接读取图像并保存到session_state
            image = Image.open(uploaded_file)
            image = np.array(image)
            st.session_state.extracted_frames = [image]
            st.subheader("原始图像")
            st.image(image, caption="原始图像", use_column_width=True)

        # 如果有可用图像
        if hasattr(st.session_state, 'extracted_frames') and st.session_state.extracted_frames:
            st.subheader("图像增强设置")

            # 选择要处理的图像（如果有多帧）
            if len(st.session_state.extracted_frames) > 1:
                frame_idx = st.selectbox("选择要增强的帧", range(len(st.session_state.extracted_frames)),
                                         format_func=lambda x: f"帧 {x + 1}")
                st.session_state.selected_image = st.session_state.extracted_frames[frame_idx]
            else:
                st.session_state.selected_image = st.session_state.extracted_frames[0]

            # 选择增强方法
            enhancement_methods = [
                "亮度增强", "亮度降低", "高斯模糊", "高斯噪声",
                "模拟遮挡", "旋转（逆时针45度）", "水平翻转", "垂直翻转"
            ]

            selected_methods = st.multiselect(
                "选择图像增强方法（可多选）",
                enhancement_methods,
                default=["亮度增强"]
            )

            # 遮挡尺寸设置（仅当选择了模拟遮挡时显示）
            occlusion_size = (50, 50)  # 默认值
            if "模拟遮挡" in selected_methods:
                with st.expander("调整遮挡参数", expanded=True):
                    col_w, col_h = st.columns(2)
                    with col_w:
                        occlude_width = st.slider("遮挡块宽度", 10, 300, 50)
                    with col_h:
                        occlude_height = st.slider("遮挡块高度", 10, 300, 50)
                    occlusion_size = (occlude_width, occlude_height)

            # 执行增强并保存结果到session_state
            if st.button("执行图像增强") and selected_methods:
                with st.spinner("正在进行图像增强..."):
                    enhanced_images, method_names = apply_enhancements(
                        st.session_state.selected_image,
                        selected_methods,
                        occlusion_size=occlusion_size  # 传入自定义遮挡尺寸
                    )
                    st.session_state.enhanced_results = enhanced_images
                    st.session_state.enhanced_names = method_names

            # 显示增强结果
            if hasattr(st.session_state, 'enhanced_results') and st.session_state.enhanced_results:
                st.subheader("增强结果预览")

                # 显示原始图像
                st.image(st.session_state.selected_image, caption="原始图像", use_column_width=True)

                # 显示增强结果
                cols = st.columns(2)
                for i, (enhanced_img, method_name) in enumerate(
                        zip(st.session_state.enhanced_results, st.session_state.enhanced_names)):
                    with cols[i % 2]:
                        st.image(enhanced_img, caption=f"{method_name}", use_column_width=True)

                        # 单个下载
                        img_bytes = image_to_bytes(enhanced_img)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label=f"下载 {method_name}",
                            data=img_bytes,
                            file_name=f"enhanced_{method_name}_{timestamp}.png",
                            mime="image/png",
                            key=f"download_{i}"
                        )

                # 全部下载（ZIP）
                zip_buffer = create_zip(
                    st.session_state.enhanced_results,
                    [f"enhanced_{name}" for name in st.session_state.enhanced_names]
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="下载所有增强结果 (ZIP)",
                    data=zip_buffer,
                    file_name=f"enhanced_results_{timestamp}.zip",
                    mime="application/zip"
                )


if __name__ == "__main__":
    main()