import os
import uuid
import time
from pathlib import Path
import streamlit as st
from src.application.rag_pipeline import RAGPipeline
from src.domain.services.retrieval_config import RetrievalConfig
from src.domain.exceptions import (
    IngestionError,
    DuplicateFileError,
    SecurityValidationError,
    CorruptedFileError,
    FileTooLargeError,
    UnsupportedFileError
)
from src.utils.logger import logger

# 1. Cấu hình trang Streamlit
st.set_page_config(
    page_title="NLP RAG Chatbot Space",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Định nghĩa thư mục lưu trữ gốc
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage" / "workspaces"

# Đảm bảo thư mục lưu trữ gốc của các workspace tồn tại
os.makedirs(STORAGE_DIR, exist_ok=True)

# 3. Các hàm bổ trợ quét Workspace cục bộ
def get_existing_workspaces():
    """Quét thư mục storage và trả về danh sách các UUID hợp lệ."""
    if not STORAGE_DIR.exists():
        return []
    workspaces = []
    for item in STORAGE_DIR.iterdir():
        if item.is_dir():
            try:
                # Kiểm tra xem tên thư mục có phải UUID hợp lệ không
                uuid.UUID(item.name)
                workspaces.append(item.name)
            except ValueError:
                pass
    return sorted(workspaces)

# 4. CSS Custom tối ưu không gian và tạo hiệu ứng kính mờ (Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Thiết lập sidebar gọn gàng */
    [data-testid="stSidebar"] {
        background-color: #0B0F19;
    }
    
    /* Hộp UUID hiện tại */
    .current-uuid-label {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .current-uuid-value {
        background-color: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.2);
        color: #38BDF8;
        border-radius: 8px;
        padding: 8px 12px;
        font-family: monospace;
        font-size: 0.85rem;
        word-break: break-all;
        margin-bottom: 12px;
        text-align: center;
    }
    
    /* Item danh sách file compact */
    .compact-file-item {
        background-color: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.85rem;
        color: #E2E8F0;
        display: flex;
        align-items: center;
        gap: 6px;
        word-break: break-all;
    }
    
    .compact-file-size {
        font-size: 0.7rem;
        color: #64748B;
    }
    
    /* Container của Chat và Scroll */
    .chat-container {
        padding: 10px;
        border-radius: 16px;
        background-color: rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# 5. Khởi động RAG Pipeline (Singleton via st.cache_resource)
@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()

try:
    rag_pipeline = get_rag_pipeline()
except Exception as e:
    st.error(f"Lỗi khởi động RAG Pipeline (Hãy kiểm tra cấu hình hoặc thư viện): {str(e)}")
    st.stop()

# 6. Khởi tạo trạng thái phiên làm việc (Session States)
if "workspace_id" not in st.session_state:
    st.session_state.workspace_id = str(uuid.uuid4())

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant", 
            "content": "Chào mừng bạn đến với Không gian Nghiên cứu NLP! 🏛️\nHãy tải các tài liệu PDF hoặc DOCX của bạn lên ở cột bên trái, sau đó đặt câu hỏi tại đây để thảo luận và truy xuất dữ liệu ngữ nghĩa từ Vector Space.",
            "diagnostics": None,
            "sources": None
        }
    ]

current_ws_id = st.session_state.workspace_id
ws_dir = STORAGE_DIR / current_ws_id / "pdfs"
os.makedirs(ws_dir, exist_ok=True)

# ==========================================
# SIDEBAR: QUẢN LÝ KHÔNG GIAN (WORKSPACE)
# ==========================================
with st.sidebar:
    st.markdown("<h3 style='color: #38BDF8; font-family: \"Outfit\", sans-serif; font-weight: 700; margin-top: -15px; margin-bottom: 15px;'>🏛️ NLP Workspace</h3>", unsafe_allow_html=True)
    
    # 1. Đẩy UUID không gian hiện tại lên cao nhất
    st.markdown("<div class='current-uuid-label'>WORKSPACE HIỆN TẠI:</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='current-uuid-value'>{current_ws_id}</div>", unsafe_allow_html=True)
    
    # 2. Ngay bên dưới là nút Tạo không gian mới
    if st.button("✨ Tạo Không gian mới", use_container_width=True, type="secondary"):
        st.session_state.workspace_id = str(uuid.uuid4())
        st.session_state.chat_history = [
            {
                "role": "assistant", 
                "content": "Chào mừng bạn đến với Không gian Nghiên cứu NLP mới! 🏛️\nHãy tải các tài liệu PDF hoặc DOCX của bạn lên ở cột bên trái, sau đó đặt câu hỏi tại đây.",
                "diagnostics": None,
                "sources": None
            }
        ]
        st.rerun()
        
    st.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    
    # 3. Mục Cấu hình OpenAI API Key (Task C.1)
    st.markdown("<div class='current-uuid-label'>🔑 CẤU HÌNH OPENAI API KEY:</div>", unsafe_allow_html=True)
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key:",
        value=st.session_state.openai_api_key,
        type="password",
        placeholder="Nhập sk-...",
        label_visibility="collapsed",
        help="Cung cấp API Key để tổng hợp câu trả lời tự động. Dữ liệu chỉ được lưu trữ tạm thời trong session state trình duyệt."
    )
    
    st.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    
    # 4. Danh sách các Workspace cũ (Quét động dưới đĩa)
    st.markdown("<h4 style='color: #E2E8F0; font-family: \"Outfit\", sans-serif; font-weight: 600; margin-top: 15px; margin-bottom: 10px;'>🔄 Danh sách Không gian</h4>", unsafe_allow_html=True)
    existing_ws = get_existing_workspaces()
    
    if not existing_ws:
        st.info("Chưa tìm thấy workspace cũ nào trên hệ thống.")
    else:
        # Tạo một scrollable container với chiều cao cố định để tự động hiển thị thanh cuộn khi vượt quá giới hạn
        with st.container(height=240, border=False):
            for ws in existing_ws:
                is_current = (ws == current_ws_id)
                btn_type = "primary" if is_current else "secondary"
                short_name = f"📍 {ws[:8]}...{ws[-4:]}" if is_current else f"📁 {ws[:8]}...{ws[-4:]}"
                
                if st.button(short_name, key=f"ws_btn_{ws}", use_container_width=True, type=btn_type, help=f"Khôi phục Workspace: {ws}"):
                    if ws != current_ws_id:
                        st.session_state.workspace_id = ws
                        st.session_state.chat_history = [
                            {
                                "role": "assistant", 
                                "content": f"Đã khôi phục thành công Workspace: {ws} 🔄\nTài liệu của bạn đã được tải lại tại cột bên trái.",
                                "diagnostics": None,
                                "sources": None
                            }
                        ]
                        st.rerun()

# ==========================================
# TRANG CHÍNH: CHIA CỘT 2-8 LAYOUT
# ==========================================
col_sources, col_chat = st.columns([2, 8], gap="medium")

# ------------------------------------------
# CỘT NHỎ (20%): QUẢN LÝ TÀI LIỆU NGUỒN (PDF, DOCX)
# ------------------------------------------
with col_sources:
    st.markdown("#### 📂 Tài liệu Nguồn")
    
    # Nút Popover ẩn khung Upload để tiết kiệm diện tích tối đa
    with st.popover("📤 Tải lên tài liệu mới", use_container_width=True):
        st.markdown("**Chọn tệp PDF hoặc DOCX từ máy tính:**")
        uploaded_files = st.file_uploader(
            "Tải lên tệp PDF/DOCX:",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            if st.button("💾 Lưu & Nạp Vào Vector Space", type="primary", use_container_width=True):
                saved_count = 0
                ingested_count = 0
                for uploaded_file in uploaded_files:
                    target_path = ws_dir / uploaded_file.name
                    try:
                        # 1. Lưu vật lý xuống ổ đĩa
                        with open(target_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_count += 1
                        
                        # 2. Thực hiện nạp semantic embedding tự động
                        with st.spinner(f"Đang bóc tách và tạo Vector nhúng cho tệp {uploaded_file.name}..."):
                            status = rag_pipeline.ingest_document(
                                workspace_id=current_ws_id, 
                                file_path=str(target_path), 
                                filename=uploaded_file.name,
                                openai_api_key=st.session_state.openai_api_key
                            )
                            if status == "SUCCESS":
                                ingested_count += 1
                            else:
                                st.warning(f"{uploaded_file.name}: {status}")
                    except DuplicateFileError as e:
                        logger.warning(f"Bypass: {str(e)}")
                        st.toast(str(e), icon="⚠️")
                    except FileTooLargeError as e:
                        st.error(f"❌ {uploaded_file.name}: Tệp quá dung lượng cho phép. {str(e)}")
                    except SecurityValidationError as e:
                        st.error(f"🛡️ Cảnh báo Bảo mật: {uploaded_file.name} bị từ chối. {str(e)}")
                    except CorruptedFileError as e:
                        st.error(f"⚠️ Lỗi cấu trúc: Tệp {uploaded_file.name} bị hỏng. {str(e)}")
                    except Exception as e:
                        st.error(f"Lỗi khi xử lý {uploaded_file.name}: {str(e)}")
                
                if saved_count > 0:
                    if ingested_count > 0:
                        st.toast(f"Đã lưu thành công {saved_count} tệp! ({ingested_count} tệp được nạp vào Qdrant)", icon="⚡")
                    st.rerun()
                    
    # Quét danh sách file đã nạp từ RAG pipeline (Task C.2)
    db_files = rag_pipeline.get_workspace_documents(current_ws_id)
        
    # Hiển thị danh sách file compact kèm Status Badge trực quan
    if not db_files:
        st.markdown("<span style='font-size: 0.8rem; color: #64748B;'>Chưa có tài liệu nào trong không gian này.</span>", unsafe_allow_html=True)
    else:
        for idx, doc_data in enumerate(db_files):
            filename = doc_data["filename"]
            status = doc_data["status"]
            chunks = doc_data["chunk_count"]
            err = doc_data.get("error_message")
            
            file_path = ws_dir / filename
            size_str = "Chưa lưu"
            if file_path.exists():
                size_kb = os.path.getsize(file_path) / 1024
                size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{(size_kb/1024):.2f} MB"
                
            perf = doc_data.get("performance_metrics")
            latency_str = ""
            if perf and "total_ingestion" in perf:
                latency_str = f" • {perf['total_ingestion']:.0f}ms"
                
            # Khởi tạo badge động theo trạng thái của Pipeline
            status_badge = "⚡" if status == "COMPLETED" else ("⏳" if status == "PROCESSING" else "❌")
            status_desc = f"{chunks} Chunks{latency_str}" if status == "COMPLETED" else ("Đang nạp..." if status == "PROCESSING" else "Nạp lỗi")
            status_border_color = "#10B981" if status == "COMPLETED" else ("#F59E0B" if status == "PROCESSING" else "#EF4444")
            
            # Giao diện compact nhỏ gọn cho danh sách file
            st.markdown(f"""
            <div class='compact-file-item' style='border-left: 3px solid {status_border_color};'>
                <span style='font-size: 1.15rem;'>{status_badge}</span>
                <div style='flex-grow: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>
                    <span title='{filename}' style='font-weight: 500;'>{filename}</span><br>
                    <span class='compact-file-size'>{size_str} • {status_desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if err:
                st.caption(f":red[Lỗi: {err}]")
            
        # Nút dọn dẹp nhỏ gọn
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Xóa sạch nguồn", use_container_width=True):
            with st.spinner("Đang xóa dữ liệu Vector Space và tệp cục bộ..."):
                rag_pipeline.clear_workspace(current_ws_id)
            st.toast("Đã dọn dẹp sạch toàn bộ không gian làm việc!", icon="🗑️")
            st.rerun()

# ------------------------------------------
# CỘT LỚN (80%): KHUNG TABS (CHAT & BENCHMARK)
# ------------------------------------------
with col_chat:
    tab_chat, tab_bench = st.tabs(["💬 Thảo luận RAG Engine", "📊 So sánh & Kiểm chuẩn (Benchmarking)"])
    
    # -------------------------------------------------------------
    # TAB 1: THẢO LUẬN HỘI THOẠI RAG
    # -------------------------------------------------------------
    with tab_chat:
        st.markdown("#### 💬 Thảo luận Tài liệu & NLP RAG Engine")
        
        # Hiển thị luồng lịch sử chat động kèm Diagnostics Viewer (Task C.4)
        chat_placeholder = st.container()
        with chat_placeholder:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
                    # Hiển thị chẩn đoán truy hồi cũ nếu có lưu trong lịch sử
                    if msg.get("role") == "assistant" and msg.get("diagnostics"):
                        diag = msg["diagnostics"]
                        sources = msg["sources"]
                        
                        st.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px dashed rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
                        
                        # Bảng chẩn đoán
                        latency_rows = ""
                        if "vector_search" in diag:
                            latency_rows += f"""
<tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
  <td style='padding: 4px; color: #64748B; padding-left: 12px;'>└─ Vector Store Search Time</td>
  <td style='text-align: right; padding: 4px; color: #38BDF8; font-family: monospace;'>{diag["vector_search"]:.1f} ms</td>
</tr>
"""
                        if "llm_synthesis" in diag:
                            latency_rows += f"""
<tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
  <td style='padding: 4px; color: #64748B; padding-left: 12px;'>└─ OpenAI API Synthesis Time</td>
  <td style='text-align: right; padding: 4px; color: #38BDF8; font-family: monospace;'>{diag["llm_synthesis"]:.1f} ms</td>
</tr>
"""
                        if "total_query" in diag:
                            latency_rows += f"""
<tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
  <td style='padding: 4px; color: #E2E8F0; font-weight: 500;'>Total RAG Pipeline Latency</td>
  <td style='text-align: right; padding: 4px; color: #10B981; font-weight: 500; font-family: monospace;'>{diag["total_query"]:.1f} ms</td>
</tr>
"""

                        diag_html = f"""
<table style='width:100%; font-size: 0.8rem; border-collapse: collapse; margin-bottom: 10px;'>
  <tr style='border-bottom: 1px solid rgba(255,255,255,0.05); color: #64748B;'>
    <th style='text-align: left; padding: 4px;'>Chỉ số Chẩn đoán (Metric)</th>
    <th style='text-align: right; padding: 4px;'>Giá trị (Value)</th>
  </tr>
  <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
    <td style='padding: 4px; color: #E2E8F0;'>Query Embedding Dimension</td>
    <td style='text-align: right; padding: 4px; color: #38BDF8; font-family: monospace;'>{diag["query_dimension"]}</td>
  </tr>
  <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
    <td style='padding: 4px; color: #E2E8F0;'>Max Similarity Score (Cosine)</td>
    <td style='text-align: right; padding: 4px; color: #10B981; font-family: monospace;'>{diag["max_score"]:.4f}</td>
  </tr>
  <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
    <td style='padding: 4px; color: #E2E8F0;'>Average Similarity Score</td>
    <td style='text-align: right; padding: 4px; color: #F59E0B; font-family: monospace;'>{diag["avg_score"]:.4f}</td>
  </tr>
  <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
    <td style='padding: 4px; color: #E2E8F0;'>Retrieved Semantic Chunks</td>
    <td style='text-align: right; padding: 4px; color: #A855F7; font-family: monospace;'>{diag["retrieved_count"]} Chunks</td>
  </tr>
  {latency_rows}
</table>
"""
                        
                        # Hộp mở rộng chẩn đoán chứa cả Latency Spans và các đoạn văn bản truy xuất
                        with st.expander("⚙️ Báo cáo chẩn đoán & Nguồn truy xuất (Diagnostics & Latency Spans)"):
                            st.markdown(diag_html, unsafe_allow_html=True)
                            st.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px dashed rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
                            st.markdown("<div style='font-weight: 600; font-size: 0.85rem; color: #64748B; margin-bottom: 8px;'>📄 CÁC ĐOẠN VĂN BẢN TRUY XUẤT (RETIRIEVED CHUNKS):</div>", unsafe_allow_html=True)
                            for s_idx, src in enumerate(sources):
                                st.markdown(f"""
                                <div style='background-color: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 10px; margin-bottom: 8px;'>
                                    <div style='display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748B; margin-bottom: 4px;'>
                                        <span>📄 Đoạn {s_idx + 1} - Trang {src["page_number"]}</span>
                                        <span style='color: #10B981;'>Độ khớp: {(src["score"]*100):.1f}%</span>
                                    </div>
                                    <div style='font-size: 0.85rem; color: #E2E8F0; line-height: 1.4; font-style: italic;'>
                                        "{src["text"]}"
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
        # Xử lý nhập tin nhắn từ người dùng
        if user_query := st.chat_input("Hỏi tôi bất cứ điều gì về tài liệu của bạn..."):
            # 1. Hiển thị tin nhắn người dùng
            with st.chat_message("user"):
                st.markdown(user_query)
                
            # Lưu vào lịch sử
            st.session_state.chat_history.append({
                "role": "user", 
                "content": user_query,
                "diagnostics": None,
                "sources": None
            })
            
            # 2. Xử lý phản hồi RAG thực tế từ Pipeline (Task C.3)
            with st.chat_message("assistant"):
                with st.spinner("Đang truy xuất Vector Space và tổng hợp câu trả lời..."):
                    # Default queries run on baseline path
                    rag_res = rag_pipeline.query_workspace(
                        workspace_id=current_ws_id,
                        question=user_query,
                        openai_api_key=st.session_state.openai_api_key,
                        routing_path="baseline"
                    )
                    
                    response_text = rag_res["answer"]
                    diagnostics = rag_res["diagnostics"]
                    sources = rag_res["sources"]
                    
                    st.markdown(response_text)
                    
                    # Hiển thị chẩn đoán truy hồi và viewer
                    if diagnostics["retrieved_count"] > 0:
                        st.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px dashed rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
                        
                        # Build execution latency rows if present in Diagnostics metrics
                        latency_rows = ""
                        if "vector_search" in diagnostics:
                            latency_rows += f"""
<tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
  <td style='padding: 4px; color: #64748B; padding-left: 12px;'>└─ Vector Store Search Time</td>
  <td style='text-align: right; padding: 4px; color: #38BDF8; font-family: monospace;'>{diagnostics["vector_search"]:.1f} ms</td>
</tr>
"""
                        if "llm_synthesis" in diagnostics:
                            latency_rows += f"""
<tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
  <td style='padding: 4px; color: #64748B; padding-left: 12px;'>└─ OpenAI API Synthesis Time</td>
  <td style='text-align: right; padding: 4px; color: #38BDF8; font-family: monospace;'>{diagnostics["llm_synthesis"]:.1f} ms</td>
</tr>
"""
                        if "total_query" in diagnostics:
                            latency_rows += f"""
<tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
  <td style='padding: 4px; color: #E2E8F0; font-weight: 500;'>Total RAG Pipeline Latency</td>
  <td style='text-align: right; padding: 4px; color: #10B981; font-weight: 500; font-family: monospace;'>{diagnostics["total_query"]:.1f} ms</td>
</tr>
"""

                        diag_html = f"""
<table style='width:100%; font-size: 0.8rem; border-collapse: collapse; margin-bottom: 10px;'>
  <tr style='border-bottom: 1px solid rgba(255,255,255,0.05); color: #64748B;'>
    <th style='text-align: left; padding: 4px;'>Chỉ số Chẩn đoán (Metric)</th>
    <th style='text-align: right; padding: 4px;'>Giá trị (Value)</th>
  </tr>
  <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
    <td style='padding: 4px; color: #E2E8F0;'>Query Embedding Dimension</td>
    <td style='text-align: right; padding: 4px; color: #38BDF8; font-family: monospace;'>{diagnostics["query_dimension"]}</td>
  </tr>
  <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
    <td style='padding: 4px; color: #E2E8F0;'>Max Similarity Score (Cosine)</td>
    <td style='text-align: right; padding: 4px; color: #10B981; font-family: monospace;'>{diagnostics["max_score"]:.4f}</td>
  </tr>
  <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
    <td style='padding: 4px; color: #E2E8F0;'>Average Similarity Score</td>
    <td style='text-align: right; padding: 4px; color: #F59E0B; font-family: monospace;'>{diagnostics["avg_score"]:.4f}</td>
  </tr>
  <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
    <td style='padding: 4px; color: #E2E8F0;'>Retrieved Semantic Chunks</td>
    <td style='text-align: right; padding: 4px; color: #A855F7; font-family: monospace;'>{diagnostics["retrieved_count"]} Chunks</td>
  </tr>
  {latency_rows}
</table>
"""
                        
                        # Hộp mở rộng chẩn đoán chứa cả Latency Spans và các đoạn văn bản truy xuất
                        with st.expander("⚙️ Báo cáo chẩn đoán & Nguồn truy xuất (Diagnostics & Latency Spans)"):
                            st.markdown(diag_html, unsafe_allow_html=True)
                            st.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px dashed rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
                            st.markdown("<div style='font-weight: 600; font-size: 0.85rem; color: #64748B; margin-bottom: 8px;'>📄 CÁC ĐOẠN VĂN BẢN TRUY XUẤT (RETIRIEVED CHUNKS):</div>", unsafe_allow_html=True)
                            for s_idx, src in enumerate(sources):
                                st.markdown(f"""
                                <div style='background-color: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 10px; margin-bottom: 8px;'>
                                    <div style='display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748B; margin-bottom: 4px;'>
                                        <span>📄 Đoạn {s_idx + 1} - Trang {src["page_number"]}</span>
                                        <span style='color: #10B981;'>Độ khớp: {(src["score"]*100):.1f}%</span>
                                    </div>
                                    <div style='font-size: 0.85rem; color: #E2E8F0; line-height: 1.4; font-style: italic;'>
                                        "{src["text"]}"
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
            # Lưu vào lịch sử
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_text,
                "diagnostics": diagnostics if diagnostics["retrieved_count"] > 0 else None,
                "sources": sources if diagnostics["retrieved_count"] > 0 else None
            })
            st.rerun()

    # -------------------------------------------------------------
    # TAB 2: BẢNG SO SÁNH & KIỂM CHUẨN (BENCHMARKING)
    # -------------------------------------------------------------
    with tab_bench:
        st.markdown("<h4 style='color: #38BDF8; font-family: \"Outfit\", sans-serif;'>📊 Bộ khung So sánh & Kiểm chuẩn Đa Mô hình (Benchmarking Harness)</h4>", unsafe_allow_html=True)
        st.markdown("""
        Bảng điều khiển này cho phép chạy thử nghiệm đồng thời **5 câu hỏi kiểm chuẩn** (ground-truth) lưu tại [questions.json](file:///Users/macos/SDK/evaluation/questions.json) trên tài liệu trong Workspace hiện tại 
        để so sánh đối xứng hiệu năng và độ chính xác của **3 hướng truy hồi chuyên biệt** và hiệu năng vector hóa.
        """)
        
        db_files = rag_pipeline.get_workspace_documents(current_ws_id)
        completed_files = [f for f in db_files if f["status"] == "COMPLETED"]
        
        if not completed_files:
            st.info("⚠️ Vui lòng tải lên và nạp thành công ít nhất một tài liệu PDF hoặc DOCX ở cột bên trái trước khi bắt đầu kiểm chuẩn.")
        else:
            import pandas as pd
            import json
            from datetime import datetime
            
            benchmark_history_path = STORAGE_DIR / current_ws_id / "benchmark_history.json"
            ingestion_history_path = STORAGE_DIR / current_ws_id / "ingestion_history.json"
            
            # --- PHẦN 1: KIỂM CHUẨN LẬP CHỈ MỤC VECTOR (INGESTION BENCHMARKS) ---
            st.markdown("<h5 style='color: #10B981; font-family: \"Outfit\", sans-serif; margin-top: 20px;'>⚡ 1. Hiệu năng Vector hoá & Lập chỉ mục (Ingestion Benchmarks)</h5>", unsafe_allow_html=True)
            st.markdown("""
            Bảng dưới đây so sánh thời gian nhúng và chèn vector (Vectorization & Indexing Latency) của **3 mô hình** trên các tài liệu đã nạp thành công vào Workspace này.
            """)
            
            if ingestion_history_path.exists():
                try:
                    with open(ingestion_history_path, "r", encoding="utf-8") as f:
                        ing_history = json.load(f)
                    
                    if ing_history:
                        ing_rows = []
                        chart_ing_rows = []
                        for idx, run in enumerate(ing_history):
                            fn = run.get("filename", f"Tài liệu {idx+1}")
                            chunks = run.get("chunk_count", 0)
                            t_minilm = run.get("indexing_minilm_ms", 0.0)
                            t_bge = run.get("indexing_bge_ms", 0.0)
                            t_openai = run.get("indexing_openai_ms", 0.0)
                            total_t = run.get("total_ingestion_ms", 0.0)
                            
                            ing_rows.append({
                                "Tên tài liệu": fn,
                                "Số đoạn (Chunks)": chunks,
                                "MiniLM (Local - 384d)": f"{t_minilm:.1f} ms" if t_minilm > 0 else "N/A",
                                "BGE-Small (Local - 384d)": f"{t_bge:.1f} ms" if t_bge > 0 else "N/A",
                                "OpenAI (Cloud - 1536d)": f"{t_openai:.1f} ms" if t_openai > 0 else "Vô hiệu (Thiếu OpenAI Key)",
                                "Tổng thời gian nạp": f"{total_t:.1f} ms"
                            })
                            
                            if t_minilm > 0:
                                chart_ing_rows.append({"Tài liệu": fn, "Mô hình": "MiniLM (Local)", "Thời gian (ms)": t_minilm})
                            if t_bge > 0:
                                chart_ing_rows.append({"Tài liệu": fn, "Mô hình": "BGE-Small (Local)", "Thời gian (ms)": t_bge})
                            if t_openai > 0:
                                chart_ing_rows.append({"Tài liệu": fn, "Mô hình": "OpenAI (Cloud)", "Thời gian (ms)": t_openai})
                        
                        st.dataframe(pd.DataFrame(ing_rows), use_container_width=True, hide_index=True)
                        
                        if chart_ing_rows:
                            st.markdown("###### ⏱️ So sánh Thời gian Vector hóa giữa các Mô hình (ms)")
                            chart_ing_df = pd.DataFrame(chart_ing_rows)
                            import altair as alt
                            # Clustered bar chart
                            chart = alt.Chart(chart_ing_df).mark_bar().encode(
                                x=alt.X('Mô hình:N', title=None),
                                y=alt.Y('Thời gian (ms):Q', title='Thời gian (ms)'),
                                color=alt.Color('Mô hình:N', scale=alt.Scale(range=['#38BDF8', '#10B981', '#F59E0B'])),
                                column=alt.Column('Tài liệu:N', title=None)
                            ).properties(width=160, height=180)
                            st.altair_chart(chart, use_container_width=True)
                        
                        # --- PHASE 4: Chi phí trung bình ms/chunk cho mỗi mô hình ---
                        st.markdown("###### 📐 Hiệu suất Vector hóa trung bình (ms/chunk)")
                        cost_rows = []
                        for run in ing_history:
                            fn = run.get("filename", "N/A")
                            chunks = run.get("chunk_count", 0)
                            if chunks <= 0:
                                continue
                            t_minilm = run.get("indexing_minilm_ms", 0.0)
                            t_bge = run.get("indexing_bge_ms", 0.0)
                            t_openai = run.get("indexing_openai_ms", 0.0)
                            if t_minilm > 0:
                                cost_rows.append({"Tài liệu": fn, "Mô hình": "MiniLM (Local)", "ms/chunk": round(t_minilm / chunks, 2)})
                            if t_bge > 0:
                                cost_rows.append({"Tài liệu": fn, "Mô hình": "BGE-Small (Local)", "ms/chunk": round(t_bge / chunks, 2)})
                            if t_openai > 0:
                                cost_rows.append({"Tài liệu": fn, "Mô hình": "OpenAI (Cloud)", "ms/chunk": round(t_openai / chunks, 2)})
                        
                        if cost_rows:
                            cost_df = pd.DataFrame(cost_rows)
                            cost_chart = alt.Chart(cost_df).mark_bar(
                                cornerRadiusTopLeft=4,
                                cornerRadiusTopRight=4
                            ).encode(
                                x=alt.X('Mô hình:N', title=None, axis=alt.Axis(labelAngle=0)),
                                y=alt.Y('ms/chunk:Q', title='ms / chunk'),
                                color=alt.Color('Mô hình:N', scale=alt.Scale(
                                    domain=['MiniLM (Local)', 'BGE-Small (Local)', 'OpenAI (Cloud)'],
                                    range=['#38BDF8', '#10B981', '#F59E0B']
                                ), legend=None),
                                column=alt.Column('Tài liệu:N', title=None),
                                tooltip=[
                                    alt.Tooltip('Tài liệu:N'),
                                    alt.Tooltip('Mô hình:N'),
                                    alt.Tooltip('ms/chunk:Q', format='.2f')
                                ]
                            ).properties(width=140, height=160)
                            st.altair_chart(cost_chart, use_container_width=True)
                        else:
                            st.caption("Không đủ dữ liệu chunk để tính ms/chunk.")
                    else:
                        st.info("Chưa có lịch sử vector hóa cho workspace này.")
                except Exception as e:
                    st.error(f"Lỗi đọc lịch sử vector hóa: {str(e)}")
            else:
                st.info("💡 Chưa ghi nhận lịch sử vector hóa. Hãy thử tải lên tệp tin ở Sidebar để lưu lại metric tự động!")
                
            st.markdown("<hr style='border-top: 1px solid #1E293B; margin: 25px 0;'/>", unsafe_allow_html=True)
            
            # --- PHẦN 2: KIỂM CHUẨN TRUY HỒI RAG (RETRIEVAL BENCHMARKS) ---
            st.markdown("<h5 style='color: #38BDF8; font-family: \"Outfit\", sans-serif;'>📊 2. Bộ khung So sánh & Kiểm chuẩn Truy hồi RAG (Retrieval Benchmarking)</h5>", unsafe_allow_html=True)
            st.markdown("""
            Đánh giá chất lượng của **4 hướng truy hồi chuyên biệt** qua bộ 5 câu hỏi ground-truth:
            1. **EXACT (BM25 Lexical)**: Truy hồi từ khóa thô (Sparse Retrieval) dựa trên tần suất từ (TF-IDF cải tiến).
            2. **BASELINE (MiniLM Local)**: Standard Dense Retrieval sử dụng mô hình nhúng cục bộ `MiniLM` và cấu hình HNSW mặc định.
            3. **ADVANCED (BGE Local)**: High-Capacity Local Dense sử dụng mô hình `BGE-Small` và HNSW graph parameters tinh chỉnh nâng cao (`ef_search=64`).
            4. **OPENAI CLOUD**: Production Standard sử dụng mô hình nhúng OpenAI Cloud `text-embedding-3-small` (1536 chiều) và thuật toán Cosine Reranking.
            """)
            
            key_to_use = st.session_state.openai_api_key
            if not (key_to_use and key_to_use.strip().startswith("sk-")):
                st.warning("⚠️ **Lưu ý**: OpenAI API Key chưa được cấu hình ở sidebar. Bạn vẫn có thể chạy benchmark, nhưng Hướng 3 (OpenAI Cloud) sẽ tự động bị bỏ qua.")
            
            # Load latest benchmark run
            latest_results = None
            history = []
            if benchmark_history_path.exists():
                try:
                    with open(benchmark_history_path, "r", encoding="utf-8") as f:
                        history = json.load(f)
                    if history:
                        latest_results = history[-1]
                except Exception as e:
                    st.error(f"Lỗi đọc lịch sử kiểm chuẩn: {str(e)}")
            
            # Nút chạy benchmark mới
            if st.button("⚡ Bắt đầu Chạy Benchmark Đa Mô Hình Mới", type="primary", use_container_width=True):
                from src.application.benchmark_harness import BenchmarkHarness
                harness = BenchmarkHarness(rag_pipeline)
                
                with st.spinner("Đang chạy truy hồi và tính toán metrics (Recall@K, Precision@K, MRR, Latency)..."):
                    results = harness.run_benchmark(current_ws_id, st.session_state.openai_api_key)
                    
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.success("🎉 Đã hoàn thành quá trình kiểm chuẩn đa luồng thành công!")
                    st.rerun()
            
            # Render the latest results if exist
            if latest_results:
                st.markdown("##### 🏆 Kết quả lượt Kiểm chuẩn Gần nhất (Latest Retrieval Run)")
                dt_str = datetime.fromtimestamp(latest_results["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"Phiên bản kiểm chuẩn chạy lúc: **{dt_str}**")
                
                # 1. Hiển thị Bảng Metrics đối xứng
                st.dataframe(latest_results["aggregated"], use_container_width=True, hide_index=True)
                
                # 2. Vẽ Biểu đồ so sánh Latency
                st.markdown("##### ⏱️ So sánh Độ trễ truy hồi trung bình (Avg Retrieval Latency Spans)")
                chart_rows = []
                for row in latest_results["aggregated"]:
                    path_name = row["Định hướng truy hồi (Path)"]
                    latency_str = row["Độ trễ trung bình (Avg Latency)"].replace(" ms", "")
                    try:
                        latency_val = float(latency_str)
                    except ValueError:
                        latency_val = 0.0
                    chart_rows.append({
                        "Định hướng": path_name,
                        "Avg Latency (ms)": latency_val
                    })
                
                chart_df = pd.DataFrame(chart_rows)
                st.bar_chart(chart_df, x="Định hướng", y="Avg Latency (ms)", color="#38BDF8")
                
                # 3. Hiển thị chi tiết logs của từng câu hỏi
                with st.expander("🔍 Xem chi tiết dấu vết từng lượt truy hồi (Detailed Logs Trace)"):
                    st.dataframe(latest_results["logs"], use_container_width=True, hide_index=True)
                    
                # 4. Xu hướng lịch sử nếu có nhiều lượt chạy
                if len(history) > 1:
                    with st.expander("📈 Xem xu hướng thay đổi qua các phiên bản (Historical Metrics Trends)"):
                        trend_rows = []
                        for h_idx, run in enumerate(history):
                            run_time = datetime.fromtimestamp(run["timestamp"]).strftime("%m-%d %H:%M")
                            for path_row in run["aggregated"]:
                                path_name = path_row["Định hướng truy hồi (Path)"]
                                latency_str = path_row["Độ trễ trung bình (Avg Latency)"].replace(" ms", "")
                                try:
                                    latency_val = float(latency_str)
                                except ValueError:
                                    latency_val = 0.0
                                recall_raw = path_row["Độ phủ ngữ nghĩa (Recall@K)"]
                                if recall_raw != "N/A":
                                    try:
                                        recall_val = float(recall_raw.replace("%", ""))
                                    except ValueError:
                                        recall_val = 0.0
                                else:
                                    recall_val = 0.0
                                trend_rows.append({
                                    "Lượt chạy": f"#{h_idx+1} ({run_time})",
                                    "Run": h_idx + 1,
                                    "Hướng": path_name,
                                    "Recall@K (%)": recall_val,
                                    "Latency (ms)": latency_val
                                })
                        trend_df = pd.DataFrame(trend_rows)
                        
                        if trend_df.empty:
                            st.info("Không có dữ liệu xu hướng.")
                        else:
                            # Color scale cho 4 paths
                            path_color_scale = alt.Scale(
                                domain=['BASELINE', 'ADVANCED', 'OPENAI', 'EXACT'],
                                range=['#38BDF8', '#10B981', '#F59E0B', '#A855F7']
                            )
                            
                            # --- Chart A: Recall@K Trend Line ---
                            st.markdown("###### 📈 Xu hướng Recall@K (%) qua các lượt chạy")
                            recall_line = alt.Chart(trend_df).mark_line(
                                strokeWidth=2.5,
                                point=alt.OverlayMarkDef(filled=True, size=60)
                            ).encode(
                                x=alt.X('Run:O', title='Lượt chạy (Run #)', axis=alt.Axis(
                                    labelAngle=0, labelFontSize=11, titleFontSize=12
                                )),
                                y=alt.Y('Recall@K (%):Q', title='Recall@K (%)', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(
                                    titleFontSize=12, labelFontSize=11
                                )),
                                color=alt.Color('Hướng:N', scale=path_color_scale, title='Hướng truy hồi'),
                                tooltip=[
                                    alt.Tooltip('Lượt chạy:N', title='Phiên'),
                                    alt.Tooltip('Hướng:N', title='Hướng'),
                                    alt.Tooltip('Recall@K (%):Q', format='.1f')
                                ]
                            ).properties(height=260).configure_view(
                                strokeWidth=0
                            )
                            st.altair_chart(recall_line, use_container_width=True)
                            
                            # --- Chart B: Latency Trend Line ---
                            st.markdown("###### ⏱️ Xu hướng Độ trễ truy hồi (ms) qua các lượt chạy")
                            latency_line = alt.Chart(trend_df).mark_line(
                                strokeWidth=2.5,
                                point=alt.OverlayMarkDef(filled=True, size=60)
                            ).encode(
                                x=alt.X('Run:O', title='Lượt chạy (Run #)', axis=alt.Axis(
                                    labelAngle=0, labelFontSize=11, titleFontSize=12
                                )),
                                y=alt.Y('Latency (ms):Q', title='Latency (ms)', axis=alt.Axis(
                                    titleFontSize=12, labelFontSize=11
                                )),
                                color=alt.Color('Hướng:N', scale=path_color_scale, title='Hướng truy hồi'),
                                tooltip=[
                                    alt.Tooltip('Lượt chạy:N', title='Phiên'),
                                    alt.Tooltip('Hướng:N', title='Hướng'),
                                    alt.Tooltip('Latency (ms):Q', format='.1f')
                                ]
                            ).properties(height=260).configure_view(
                                strokeWidth=0
                            )
                            st.altair_chart(latency_line, use_container_width=True)
                            
                            # Bảng dữ liệu thô bên dưới
                            with st.expander("📋 Dữ liệu bảng thô (Raw Trend Data)"):
                                st.dataframe(trend_df[['Lượt chạy', 'Hướng', 'Recall@K (%)', 'Latency (ms)']], use_container_width=True, hide_index=True)
                
                # --- PHASE 4: Tổng quan So sánh Đa chiều (Multi-Dimensional Comparison) ---
                if latest_results:
                    st.markdown("<hr style='border-top: 1px solid #1E293B; margin: 25px 0;'/>", unsafe_allow_html=True)
                    st.markdown("##### 🎯 Tổng quan So sánh Đa chiều (Multi-Dimensional Comparison)")
                    st.markdown("""
                    Biểu đồ dưới đây trực quan hóa hiệu năng tổng hợp của từng hướng truy hồi trên **4 chiều đo lường** từ lượt kiểm chuẩn gần nhất,
                    bao gồm Recall, Precision, MRR, và Tốc độ (Speed = nghịch đảo chuẩn hóa của Latency).
                    """)
                    
                    radar_rows = []
                    max_latency = 0.0
                    
                    # First pass: collect latencies to compute max for normalization
                    for row in latest_results["aggregated"]:
                        lat_str = row["Độ trễ trung bình (Avg Latency)"].replace(" ms", "")
                        try:
                            lat_val = float(lat_str)
                        except ValueError:
                            lat_val = 0.0
                        if lat_val > max_latency:
                            max_latency = lat_val
                    
                    for row in latest_results["aggregated"]:
                        path_name = row["Định hướng truy hồi (Path)"]
                        
                        # Parse recall
                        recall_raw = row["Độ phủ ngữ nghĩa (Recall@K)"]
                        try:
                            recall_pct = float(recall_raw.replace("%", "")) if recall_raw != "N/A" else 0.0
                        except ValueError:
                            recall_pct = 0.0
                        
                        # Parse precision
                        prec_raw = row["Độ chính xác (Precision@K)"]
                        try:
                            prec_pct = float(prec_raw.replace("%", "")) if prec_raw != "N/A" else 0.0
                        except ValueError:
                            prec_pct = 0.0
                        
                        # Parse MRR (0-1 scale, display as 0-100 for visual consistency)
                        mrr_raw = row["Thứ hạng nghịch đảo (MRR)"]
                        try:
                            mrr_val = float(mrr_raw) * 100.0
                        except ValueError:
                            mrr_val = 0.0
                        
                        # Compute Speed score: normalized inverse of latency (higher = faster)
                        lat_str = row["Độ trễ trung bình (Avg Latency)"].replace(" ms", "")
                        try:
                            lat_val = float(lat_str)
                        except ValueError:
                            lat_val = 0.0
                        speed_score = ((1.0 - (lat_val / max_latency)) * 100.0) if max_latency > 0 and lat_val > 0 else 0.0
                        
                        radar_rows.append({"Hướng": path_name, "Chiều đo": "Recall@K", "Điểm (%)": round(recall_pct, 1)})
                        radar_rows.append({"Hướng": path_name, "Chiều đo": "Precision@K", "Điểm (%)": round(prec_pct, 1)})
                        radar_rows.append({"Hướng": path_name, "Chiều đo": "MRR", "Điểm (%)": round(mrr_val, 1)})
                        radar_rows.append({"Hướng": path_name, "Chiều đo": "Speed", "Điểm (%)": round(speed_score, 1)})
                    
                    if radar_rows:
                        radar_df = pd.DataFrame(radar_rows)
                        
                        radar_color_scale = alt.Scale(
                            domain=['BASELINE', 'ADVANCED', 'OPENAI', 'EXACT'],
                            range=['#38BDF8', '#10B981', '#F59E0B', '#A855F7']
                        )
                        
                        radar_chart = alt.Chart(radar_df).mark_bar(
                            cornerRadiusTopRight=6,
                            cornerRadiusBottomRight=6
                        ).encode(
                            y=alt.Y('Chiều đo:N', title=None, sort=['Recall@K', 'Precision@K', 'MRR', 'Speed'], axis=alt.Axis(
                                labelFontSize=12, labelFontWeight='bold'
                            )),
                            x=alt.X('Điểm (%):Q', title='Điểm chuẩn hóa (%)', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(
                                titleFontSize=12, labelFontSize=11
                            )),
                            color=alt.Color('Hướng:N', scale=radar_color_scale, title='Hướng truy hồi'),
                            yOffset='Hướng:N',
                            tooltip=[
                                alt.Tooltip('Hướng:N', title='Hướng'),
                                alt.Tooltip('Chiều đo:N', title='Chiều đo'),
                                alt.Tooltip('Điểm (%):Q', format='.1f', title='Điểm')
                            ]
                        ).properties(height=280).configure_view(
                            strokeWidth=0
                        )
                        st.altair_chart(radar_chart, use_container_width=True)
                    else:
                        st.info("Không đủ dữ liệu để vẽ biểu đồ đa chiều.")
            else:
                st.info("💡 Chưa có dữ liệu kiểm chuẩn truy hồi. Vui lòng bấm nút **⚡ Bắt đầu Chạy Benchmark Đa Mô Hình Mới** ở trên để thực thi kiểm tra chất lượng tìm kiếm lần đầu tiên!")

            # --- PHẦN 3: ĐÁNH GIÁ CHẤT LƯỢNG BIỂU DIỄN VECTOR SPACE (REPRESENTATION QUALITY) ---
            st.markdown("<hr style='border-top: 1px solid #1E293B; margin: 25px 0;'/>", unsafe_allow_html=True)
            st.markdown("<h5 style='color: #F59E0B; font-family: \"Outfit\", sans-serif;'>💎 3. Đánh giá Chất lượng Biểu diễn Không gian Vector (Representation Space Quality)</h5>", unsafe_allow_html=True)
            st.markdown("""
            Bảng phân tích chiều sâu hình học và chất lượng cấu trúc không gian vector (**Representation Space**) của các mô hình nhúng cục bộ và đám mây. 
            Phần đánh giá này đo đạc cách mô hình sắp xếp cấu trúc tri thức của Workspace một cách tự nhiên (không cần câu hỏi truy vấn).
            """)
            
            from src.application.representation_evaluator import RepresentationEvaluator
            from src.infrastructure.embeddings.embedding_factory import EmbeddingFactory
            from src.config.settings import Settings
            import numpy as np
            
            # Helper to run representation evaluation on a collection
            def evaluate_collection_representation(col_name, provider_getter, api_key=None):
                try:
                    if "openai" in col_name and not api_key:
                        return None
                    
                    # Fetch vectors
                    vecs, doc_labels = rag_pipeline.vector_store.get_workspace_vectors(current_ws_id, col_name)
                    
                    if vecs.shape[0] < 2:
                        return {
                            "error": "Chưa đủ dữ liệu vector (Yêu cầu nạp ít nhất 2 chunks từ tài liệu để phân tích hình học).",
                            "stability_drift": None
                        }
                    
                    # 1. Clustering Silhouette
                    labels_arr = np.array(doc_labels)
                    unique_lbls = np.unique(labels_arr)
                    if len(unique_lbls) > 1:
                        silhouette = RepresentationEvaluator.calculate_silhouette_score(vecs, labels_arr)
                    else:
                        silhouette = "N/A (Cần nạp >= 2 file để tính Silhouette)"
                        
                    # 2. Isotropy (Đẳng hướng)
                    isotropy = RepresentationEvaluator.calculate_isotropy(vecs)
                    
                    # 3. Density / Intrinsic Dimensionality
                    density = RepresentationEvaluator.calculate_information_density(vecs)
                    
                    phrase_orig = "Natural Language Processing is a subfield of Artificial Intelligence."
                    phrase_mod = "Natural Language Procesaing is a subfield of Artificial Intelligence."
                    
                    try:
                        provider = provider_getter(api_key) if api_key else provider_getter()
                        emb_orig = np.array(provider.embed_query(phrase_orig))
                        emb_mod = np.array(provider.embed_query(phrase_mod))
                        drift = RepresentationEvaluator.calculate_cosine_drift(emb_orig, emb_mod)
                    except Exception:
                        drift = 0.0
                        
                    return {
                        "silhouette": silhouette,
                        "isotropy": isotropy,
                        "intrinsic_dims": density.get("intrinsic_dimensionality", 1),
                        "utilization": density.get("dimension_utilization_ratio", 0.0) * 100.0,
                        "stability_drift": drift,
                        "total_chunks": vecs.shape[0],
                        "dimensions": vecs.shape[1]
                    }
                except Exception as e:
                    return {"error": f"Lỗi tính toán: {str(e)}", "stability_drift": None}
            
            # Run evaluations
            with st.spinner("Đang phân tích cấu trúc toán học của không gian vector (SVD, Silhouette, Cosine Drift)..."):
                eval_minilm = evaluate_collection_representation(
                    rag_pipeline.vector_store.COLLECTION_MINILM, 
                    EmbeddingFactory.get_minilm_provider
                )
                eval_bge = evaluate_collection_representation(
                    rag_pipeline.vector_store.COLLECTION_BGE, 
                    EmbeddingFactory.get_bge_provider
                )
                
                # OpenAI Cloud evaluation
                openai_key = st.session_state.openai_api_key or Settings.OPENAI_API_KEY
                has_openai = bool(openai_key and openai_key.strip().startswith("sk-"))
                eval_openai = None
                if has_openai:
                    eval_openai = evaluate_collection_representation(
                        rag_pipeline.vector_store.COLLECTION_OPENAI, 
                        EmbeddingFactory.get_openai_provider,
                        openai_key
                    )
            
            # Render evaluation results in 3 clean columns
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.markdown("<h6 style='color: #38BDF8; font-family: \"Outfit\", sans-serif; font-weight: bold; margin-bottom: 10px;'>⚡ MiniLM (Local Baseline)</h6>", unsafe_allow_html=True)
                if eval_minilm and "error" not in eval_minilm:
                    st.metric("Tính Đẳng hướng (Isotropy)", f"{eval_minilm['isotropy']:.4f}", help="Mức độ phân tán đồng đều. Càng gần 1.0 càng tốt, tránh collapse không gian.")
                    
                    sil = eval_minilm['silhouette']
                    sil_str = f"{sil:.4f}" if isinstance(sil, float) else str(sil)
                    st.metric("Silhouette Score (Clustering)", sil_str, help="Chỉ số tách cụm giữa các file. Càng gần 1.0 càng tốt, thể hiện gom cụm cùng chủ đề rõ rệt.")
                    
                    st.metric("Số chiều khả dụng (PCA)", f"{eval_minilm['intrinsic_dims']} / {eval_minilm['dimensions']}", f"{eval_minilm['utilization']:.1f}%", delta_color="inverse", help="Số chiều thực tế mang 95% lượng thông tin sau SVD.")
                    
                    drift = eval_minilm['stability_drift']
                    st.metric("Độ trôi Cosine (Drift Stability)", f"{drift:.6f}", help="Mức độ trôi lệch vector dưới thay đổi văn bản cực nhỏ. Càng gần 0.0 càng tốt.")
                    st.caption(f"Tổng số phân đoạn: **{eval_minilm['total_chunks']} chunks**")
                elif eval_minilm and "error" in eval_minilm:
                    st.info(f"💡 {eval_minilm['error']}")
                else:
                    st.info("💡 Chưa có dữ liệu vector.")
                
            with col_m2:
                st.markdown("<h6 style='color: #10B981; font-family: \"Outfit\", sans-serif; font-weight: bold; margin-bottom: 10px;'>⚡ BGE-Small (Local Advanced)</h6>", unsafe_allow_html=True)
                if eval_bge and "error" not in eval_bge:
                    st.metric("Tính Đẳng hướng (Isotropy)", f"{eval_bge['isotropy']:.4f}", help="Mức độ phân tán đồng đều. Càng gần 1.0 càng tốt, tránh collapse không gian.")
                    
                    sil = eval_bge['silhouette']
                    sil_str = f"{sil:.4f}" if isinstance(sil, float) else str(sil)
                    st.metric("Silhouette Score (Clustering)", sil_str, help="Chỉ số tách cụm giữa các file. Càng gần 1.0 càng tốt, thể hiện gom cụm cùng chủ đề rõ rệt.")
                    
                    st.metric("Số chiều khả dụng (PCA)", f"{eval_bge['intrinsic_dims']} / {eval_bge['dimensions']}", f"{eval_bge['utilization']:.1f}%", delta_color="inverse", help="Số chiều thực tế mang 95% lượng thông tin sau SVD.")
                    
                    drift = eval_bge['stability_drift']
                    st.metric("Độ trôi Cosine (Drift Stability)", f"{drift:.6f}", help="Mức độ trôi lệch vector dưới thay đổi văn bản cực nhỏ. Càng gần 0.0 càng tốt.")
                    st.caption(f"Tổng số phân đoạn: **{eval_bge['total_chunks']} chunks**")
                elif eval_bge and "error" in eval_bge:
                    st.info(f"💡 {eval_bge['error']}")
                else:
                    st.info("💡 Chưa có dữ liệu vector.")
                
            with col_m3:
                st.markdown("<h6 style='color: #F59E0B; font-family: \"Outfit\", sans-serif; font-weight: bold; margin-bottom: 10px;'>⚡ OpenAI Cloud (Gold Standard)</h6>", unsafe_allow_html=True)
                if not has_openai:
                    st.info("⚪ Vô hiệu. Cần nhập OpenAI API Key ở sidebar để bật mô hình Cloud benchmark này.")
                elif eval_openai and "error" not in eval_openai:
                    st.metric("Tính Đẳng hướng (Isotropy)", f"{eval_openai['isotropy']:.4f}", help="Mức độ phân tán đồng đều. Càng gần 1.0 càng tốt, tránh collapse không gian.")
                    
                    sil = eval_openai['silhouette']
                    sil_str = f"{sil:.4f}" if isinstance(sil, float) else str(sil)
                    st.metric("Silhouette Score (Clustering)", sil_str, help="Chỉ số tách cụm giữa các file. Càng gần 1.0 càng tốt, thể hiện gom cụm cùng chủ đề rõ rệt.")
                    
                    st.metric("Số chiều khả dụng (PCA)", f"{eval_openai['intrinsic_dims']} / {eval_openai['dimensions']}", f"{eval_openai['utilization']:.1f}%", delta_color="inverse", help="Số chiều thực tế mang 95% lượng thông tin sau SVD.")
                    
                    drift = eval_openai['stability_drift']
                    st.metric("Độ trôi Cosine (Drift Stability)", f"{drift:.6f}", help="Mức độ trôi lệch vector dưới thay đổi văn bản cực nhỏ. Càng gần 0.0 càng tốt.")
                    st.caption(f"Tổng số phân đoạn: **{eval_openai['total_chunks']} chunks**")
                elif eval_openai and "error" in eval_openai:
                    st.info(f"💡 {eval_openai['error']}")
                else:
                    st.info("💡 Chưa có dữ liệu vector.")

