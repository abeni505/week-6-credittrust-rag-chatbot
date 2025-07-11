import gradio as gr
from src.rag_pipeline import initialize_components, answer_question_with_rag
import traceback

# --- 1. Initialize the RAG components once when the app starts ---
print("Application starting... Initializing RAG components.")
try:
    initialize_components()
    print("RAG components initialized successfully.")
except Exception as e:
    print(f"FATAL: Error during RAG component initialization: {e}")
    traceback.print_exc()
    exit()

# --- 2. Define the main function for the Gradio interface ---
def chatbot_interface(question, history):
    """
    This function is the core of the Gradio app. It takes the user's question
    and the chat history, then returns the formatted response.
    """
    print(f"Received question: {question}")
    
    if not question:
        return "Please ask a question."
        
    try:
        # Get the answer and sources from our RAG pipeline
        response = answer_question_with_rag(question)
        
        # --- DEBUGGING STEP ---
        # Print the full response dictionary to the terminal to check its contents.
        print("Full response from RAG pipeline:", response)
        
        # Format the output for display
        answer = response.get('answer', "No answer found.")
        sources = response.get('sources', [])
        
        # Safely get the score and format the answer string
        try:
            score = response['score']
            answer_with_score = f"{answer}\n\n*(Quality Score: {score:.4f})*"
        except KeyError:
            # This will run if the 'score' key is missing from the response
            answer_with_score = f"{answer}\n\n*(Quality Score: Not available)*"

        # Create a formatted string for the sources
        sources_text = "\n\n--- \n**Sources used to generate this answer:**\n\n"
        for i, source in enumerate(sources):
            sources_text += f"> {i+1}. \"{source}\"\n\n"
            
        return answer_with_score + sources_text
        
    except Exception as e:
        print(f"Error during RAG pipeline execution: {e}")
        traceback.print_exc()
        return "Sorry, I encountered an error while processing your request. Please check the logs for details."

# --- 3. Build and launch the Gradio Interface ---
print("Building Gradio interface...")

demo = gr.ChatInterface(
    fn=chatbot_interface,
    title="CrediTrust Complaint Analysis Chatbot 🤖",
    description="Ask questions about customer complaints. The AI will analyze the data and provide answers based on real customer feedback.",
    examples=[
        ["What are the main reasons for disputes related to money transfers?"],
        ["Why are people unhappy with their credit card rewards?"],
        ["Are there any complaints about unexpected fees on savings accounts?"]
    ],
    theme="soft"
)

if __name__ == "__main__":
    print("Launching Gradio app... Open the URL in your browser.")
    demo.launch(share=True)
