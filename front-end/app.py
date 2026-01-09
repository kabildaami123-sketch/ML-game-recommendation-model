"""
Gradio chatbot UI for recommendation system.
This module provides a web interface for the recommendation chatbot.
"""

import gradio as gr
from recommendation import get_recommendations, format_recommendations
import json


class RecommendationChatbot:
    def __init__(self):
        self.chat_history = []
    
    def chat_with_bot(self, message, mood, history):
        if not message.strip():
            return "Please enter a message."
        
        recommendations = get_recommendations(message, mood if mood != "None" else None)
        formatted_response = format_recommendations(recommendations)
        
        if mood and mood != "None":
            mood_context = f"Based on your {mood.lower()} mood:\n\n"
            formatted_response = mood_context + formatted_response
        else:
            formatted_response = "Here are my recommendations:\n\n" + formatted_response
        
        formatted_response += "\n\nWould you like different suggestions or more details?"
        
        return formatted_response

def create_chatbot_interface():
    chatbot = RecommendationChatbot()
    
    with gr.Blocks() as interface:
        gr.Markdown("# Recommendation Chatbot")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    label="Chat",
                    height=500
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="What would you like recommendations for?",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
            with gr.Column(scale=1):
                mood_dropdown = gr.Dropdown(
                    choices=["None", "Happy", "Sad", "Chill"],
                    value="None",
                    label="Mood Filter"
                )
                
                clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        def respond_to_message(message, mood, history):
            if not message.strip():
                return history, ""
            
            if history is None:
                history = []
            
            user_message = {"role": "user", "content": message}
            history.append(user_message)
            
            bot_response = chatbot.chat_with_bot(message, mood, history)
            
            bot_message = {"role": "assistant", "content": bot_response}
            history.append(bot_message)
            
            return history, ""
        
        def clear_chat():
            return [], ""
        
        def add_welcome_message():
            welcome_msg = {"role": "assistant", "content": "Hello! I can help you find movies and books. What would you like recommendations for?"}
            return [welcome_msg]
        
        send_btn.click(
            respond_to_message,
            inputs=[msg_input, mood_dropdown, chatbot_interface],
            outputs=[chatbot_interface, msg_input]
        )
        
        msg_input.submit(
            respond_to_message,
            inputs=[msg_input, mood_dropdown, chatbot_interface],
            outputs=[chatbot_interface, msg_input]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot_interface, msg_input]
        )
        
        interface.load(
            add_welcome_message,
            outputs=[chatbot_interface]
        )
    
    return interface


def main():
    """Main function to launch the chatbot interface."""
    print(" Starting Recommendation Chatbot...")
    print(" The interface will open in your browser shortly.")
    print(" Press Ctrl+C to stop the server.")
    
    interface = create_chatbot_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7862,       # Default Gradio port
        share=False,            # Set to True if you want a public link
        debug=True,             # Enable debug mode
        show_error=True         # Show errors in the interface
    )


if __name__ == "__main__":
    main()
