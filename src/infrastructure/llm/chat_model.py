from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.logger import logger

class ChatModelService:
    """Infrastructure adapter managing LangChain conversational LLM models (OpenAI/Local)."""
    
    SYSTEM_PROMPT = """Bạn là trợ lý nghiên cứu NLP học thuật xuất sắc. Nhiệm vụ của bạn là trả lời câu hỏi dựa TRÊN VÀ CHỈ DỰA TRÊN ngữ cảnh (context) được cung cấp dưới đây.

[CONTEXT]
{retrieved_context}

[RULES]
1. Chỉ trả lời dựa trên các dữ kiện có sẵn trong phần [CONTEXT].
2. KHÔNG tự ý suy đoán, suy diễn hoặc sử dụng kiến thức bên ngoài.
3. Nếu phần [CONTEXT] không cung cấp đủ thông tin để trả lời câu hỏi, hãy phản hồi chính xác câu sau:
   "Không tìm thấy thông tin phù hợp trong các tệp PDF đã tải lên." """

    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        
        # Instantiate ChatOpenAI using gpt-4o-mini (highly cost efficient, highly coherent)
        logger.info("Initializing ChatOpenAI (gpt-4o-mini, temperature=0.0)")
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=self.api_key,
            temperature=0.0
        )
        
        # Construct chat chain
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "Câu hỏi: {question}")
        ])
        
        self.chain = self.prompt_template | self.model | StrOutputParser()

    def generate_answer(self, question: str, retrieved_context: str) -> str:
        """Invokes the LLM to synthesize a grounded answer based strictly on retrieved context."""
        logger.info("Invoking LLM Chain for synthesis...")
        try:
            response = self.chain.invoke({
                "question": question,
                "retrieved_context": retrieved_context
            })
            return response
        except Exception as e:
            logger.error(f"Error during LLM invocation: {str(e)}")
            raise e
