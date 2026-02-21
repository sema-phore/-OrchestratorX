import streamlit as st
from langgraph_backend import chatBot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid
from sqlite_functions import save_thread_metadata, get_thread_metadata, get_all_thread_metadata


def generate_thread_id():
    """Generate a unique thread ID"""
    thread_id = str(uuid.uuid4())
    return thread_id


def reset_chat():
    """Reset chat and create a new thread"""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def add_thread(thread_id):
    """Add a new thread to the chat threads list"""
    if not any(thread['id'] == thread_id for thread in st.session_state['chat_threads']):
        thread = {
            "id": thread_id, 
            "name": "New Chat",
            "named": False
        }
        st.session_state['chat_threads'].append(thread)
        # Save to database
        save_thread_metadata(thread_id, "New Chat", False)


def load_conversation(thread_id):
    """Restore conversation from a thread ID using the database"""
    CONFIG = {'configurable': {'thread_id': thread_id}}
    state = chatBot.get_state(config=CONFIG)
    
    # Check if state has messages before accessing
    if state.values and 'messages' in state.values:
        return state.values['messages']
    return []


def generate_conversation_name(messages, thread_id):
    """
    Generate a name for the conversation based on message history
    Uses a separate thread ID for naming to avoid polluting the conversation history
    """
    # Use a separate thread ID for naming
    naming_thread_id = f"{thread_id}_naming"
    CONFIG = {'configurable': {'thread_id': naming_thread_id}}
    
    # Only use first 6 messages for context (3 exchanges)
    formatted_chat = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in messages[:6]
    )

    response = chatBot.invoke(
        {'messages': [HumanMessage(
            content=f"""Generate a short title (max 5 words) for this conversation.
Do not use punctuation.
Return only the title.

Conversation:
{formatted_chat}"""
        )]},
        config=CONFIG
    )

    return response['messages'][-1].content.strip()


def load_threads_from_database():
    """Load all existing threads from the SQLite database with persistent names"""
    try:
        # First, try to get all thread metadata (names are already stored)
        stored_metadata = get_all_thread_metadata()
        
        if stored_metadata:
            # Filter out naming threads and return stored metadata
            return [
                thread for thread in stored_metadata 
                if not thread['id'].endswith('_naming')
            ]
        
        # If no metadata exists (first time migration), generate from scratch
        db_thread_ids = retrieve_all_threads()
        loaded_threads = []
        
        for thread_id in db_thread_ids:
            # Skip naming threads
            if thread_id.endswith('_naming'):
                continue
            
            # Check if metadata already exists
            metadata = get_thread_metadata(thread_id)
            if metadata:
                loaded_threads.append({
                    "id": thread_id,
                    "name": metadata["name"],
                    "named": metadata["named"]
                })
                continue
            
            # Load messages for threads without metadata
            messages = load_conversation(thread_id)
            
            if messages:
                # Convert messages to temp format
                temp_messages = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        role = 'user'
                    else:
                        role = 'assistant'
                    temp_messages.append({'role': role, 'content': msg.content})
                
                # Generate name if enough messages
                if len(temp_messages) >= 3:
                    try:
                        name = generate_conversation_name(temp_messages, thread_id)
                        named = True
                    except:
                        name = temp_messages[0]['content'][:30] + "..."
                        named = False
                else:
                    name = temp_messages[0]['content'][:30] + "..." if temp_messages else "New Chat"
                    named = False
                
                # Save metadata to database
                save_thread_metadata(thread_id, name, named)
                
                loaded_threads.append({
                    "id": thread_id,
                    "name": name,
                    "named": named
                })
        
        return loaded_threads
    
    except Exception as e:
        print(f"Error loading threads from database: {e}")
        return []
