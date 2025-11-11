import React from 'react';
import { MessageSquare, Trash2, Plus } from 'lucide-react';
import './Sidebar.css';

const Sidebar = ({ 
  isOpen, 
  sessions, 
  currentSessionId, 
  onNewChat, 
  onSelectSession, 
  onDeleteSession, 
  loading 
}) => {
  return (
    <div className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
      <div className="sidebar-header">
        <button className="new-chat-btn" onClick={onNewChat}>
          <Plus size={18} />
          New Chat
        </button>
      </div>
      
      <div className="sessions-list">
        {loading ? (
          <div className="loading-sessions">Loading chats...</div>
        ) : sessions.length === 0 ? (
          <div className="no-sessions">No chat history yet</div>
        ) : (
          sessions.map((session) => (
            <div
              key={session.id}
              className={`session-item ${session.id === currentSessionId ? 'active' : ''}`}
              onClick={() => onSelectSession(session.id)}
            >
              <MessageSquare size={16} className="session-icon" />
              <span className="session-title" title={session.title}>
                {session.title}
              </span>
              <button
                className="delete-session-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteSession(session.id);
                }}
                title="Delete chat"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))
        )}
      </div>
      
      <div className="sidebar-footer">
        <div className="session-count">
          {sessions.length} {sessions.length === 1 ? 'chat' : 'chats'}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
