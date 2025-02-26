import streamlit as st
import torch
import cv2
import numpy as np
from Source.Models.Detector import VideoTransformer
from Source.Config.Config import CONFIG
import tempfile
import os


class VideoAnalyzer:
    def __init__(self, model):
        self.class_mapping = {
            'Normal': 'Bình thường',
            'Abuse': 'Lạm dụng',
            'Arrest': 'Bắt giữ',
            'Arson': 'Hỏa hoạn',
            'Assault': 'Hành hung',
            'Burglary': 'Đột nhập',
            'Explosion': 'Cháy nổ',
            'Fighting': 'Ẩu đả',
            'RoadAccidents': 'Tai nạn giao thông',
            'Robbery': 'Cướp giật',
            'Shooting': 'Nổ súng',
            'Shoplifting': 'Trộm cắp vặt',
            'Stealing': 'Trộm cắp',
            'Vandalism': 'Phá hoại'
        }

        self.model = model
        self.class_names = list(self.class_mapping.keys())  # Tên tiếng Anh để match với model
        self.vn_class_names = list(self.class_mapping.values())  # Tên tiếng Việt để hiển thị

        self.colors = {
            'Bình thường': (0, 255, 0),  # Xanh lá
            'Lạm dụng': (0, 0, 255),  # Đỏ
            'Bắt giữ': (255, 0, 0),  # Xanh dương
            'Hỏa hoạn': (0, 165, 255),  # Cam
            'Hành hung': (255, 0, 255),  # Hồng
            'Đột nhập': (255, 255, 0),  # Xanh ngọc
            'Cháy nổ': (0, 69, 255),  # Cam đỏ
            'Ẩu đả': (128, 0, 255),  # Tím đỏ
            'Tai nạn giao thông': (255, 191, 0),  # Xanh nhạt
            'Cướp giật': (0, 128, 255),  # Cam nhạt
            'Nổ súng': (255, 0, 128),  # Tím hồng
            'Trộm cắp vặt': (255, 128, 0),  # Xanh nhạt
            'Trộm cắp': (128, 128, 255),  # Hồng nhạt
            'Phá hoại': (128, 0, 0)  # Xanh đậm
        }

        self.frame_buffer = []
        self.max_confidence_class = {'class': 'Bình thường', 'confidence': 0.0}
        self.current_probabilities = {name: 0.0 for name in self.vn_class_names}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def eng_to_vn(self, eng_class):
        """Chuyển đổi tên lớp từ tiếng Anh sang tiếng Việt"""
        return self.class_mapping.get(eng_class, eng_class)

    def process_frame(self, frame):
        try:
            processed_frame = cv2.resize(frame, (64, 64))
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            self.frame_buffer.append(processed_frame)

            result = None
            if len(self.frame_buffer) == 16:
                batch = torch.from_numpy(np.stack(self.frame_buffer)).permute(3, 0, 1, 2).float() / 255.0
                batch = batch.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(batch)
                    probabilities = torch.softmax(output, dim=1)
                    class_probabilities = probabilities[0].cpu().numpy()

                    # Cập nhật xác suất cho tất cả các lớp
                    for idx, prob in enumerate(class_probabilities):
                        eng_class = self.class_names[idx]
                        vn_class = self.eng_to_vn(eng_class)
                        self.current_probabilities[vn_class] = prob

                    # Cập nhật lớp có độ tin cậy cao nhất
                    pred_class_idx = np.argmax(class_probabilities)
                    confidence = class_probabilities[pred_class_idx]
                    pred_class_eng = self.class_names[pred_class_idx]
                    pred_class_vn = self.eng_to_vn(pred_class_eng)

                    if confidence > self.max_confidence_class['confidence']:
                        self.max_confidence_class = {
                            'class': pred_class_vn,
                            'confidence': confidence
                        }

                    result = {
                        'class': pred_class_vn,
                        'confidence': confidence,
                        'is_anomaly': pred_class_eng != 'Normal',
                        'all_probabilities': self.current_probabilities
                    }

                self.frame_buffer = []

            return result

        except Exception as e:
            st.error(f"Lỗi xử lý frame: {str(e)}")
            return None


@st.cache_resource
def load_model(model_type: str):
    try:
        model_map = {
            "ViT": "vit_model.pth",
            "TimeSformer": "timesformer_model.pth",
            "Video Swin Transformer": "videoswintransformer_model.pth"
        }

        checkpoint_path = os.path.join("Checkpoints", model_map[model_type])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = VideoTransformer(
            image_size=CONFIG['data'].image_size,
            patch_size=CONFIG['model'].patch_size,
            num_classes=CONFIG['model'].num_classes,
            dim=CONFIG['model'].dim,
            depth=CONFIG['model'].depth,
            heads=CONFIG['model'].heads,
            mlp_dim=CONFIG['model'].mlp_dim,
            dropout=CONFIG['model'].dropout,
            emb_dropout=CONFIG['model'].emb_dropout,
            num_frames=CONFIG['data'].num_frames
        )

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    except Exception as e:
        st.error(f"Lỗi tải mô hình: {str(e)}")
        return None


def process_video(video_file, model):
    if model is None:
        st.error("Mô hình chưa được tải đúng cách")
        return

    try:
        analyzer = VideoAnalyzer(model)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        temp_path = tfile.name

        cap = cv2.VideoCapture(temp_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Phần tử giao diện
        video_placeholder = st.empty()
        current_analysis = st.empty()
        top_predictions = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = analyzer.process_frame(frame)

            if result:
                # Vẽ dự đoán chính
                color = analyzer.colors[result['class']]
                text = f"{result['class']}: {result['confidence']:.2%}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                if result['is_anomaly']:
                    cv2.rectangle(frame, (0, 0), (frame_width, frame_height), color, 2)

                # Sắp xếp và lấy top 5 dự đoán
                sorted_probs = sorted(
                    result['all_probabilities'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                # Hiển thị xác suất hiện tại
                current_analysis.markdown("### Phân tích hiện tại:")
                current_text = ""
                for cls, prob in sorted_probs:
                    color = analyzer.colors[cls]
                    current_text += f'<p style="color: rgb({color[2]}, {color[1]}, {color[0]})">{cls}: {prob:.2%}</p>'
                current_analysis.markdown(current_text, unsafe_allow_html=True)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()

        # Hiển thị kết quả cuối cùng
        st.success("Hoàn thành phân tích video!")

        # Hiển thị kết luận với màu sắc
        color = analyzer.colors[analyzer.max_confidence_class['class']]
        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: rgba({color[2]}, {color[1]}, {color[0]}, 0.1); border: 2px solid rgb({color[2]}, {color[1]}, {color[0]});'>
            <h2>Kết luận cuối cùng</h2>
            <h3>Hoạt động chính: {analyzer.max_confidence_class['class']}</h3>
            <h4>Độ tin cậy: {analyzer.max_confidence_class['confidence']:.2%}</h4>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Lỗi xử lý video: {str(e)}")

    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


def main():
    st.title("Hệ thống Phát hiện Bất thường trong Video")

    # Chọn mô hình
    model_type = st.selectbox(
        "Chọn mô hình",
        ["ViT", "TimeSformer", "Video Swin Transformer"]
    )

    # Tải video
    video_file = st.file_uploader("Tải lên video", type=['mp4', 'avi', 'mov'])

    if video_file:
        model = load_model(model_type)

        if model and st.button("Phân tích Video"):
            process_video(video_file, model)


if __name__ == "__main__":
    main()