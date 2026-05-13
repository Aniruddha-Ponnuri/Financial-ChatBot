#!/usr/bin/env python3

import os
import sqlite3
from typing import Any

from .logger import CustomLogger


class ChatDatabase:
    """SQLite store for chat sessions and conversation messages."""

    def __init__(self, db_path: str, logger: CustomLogger):
        self.db_path = db_path
        self.db_folder = os.path.dirname(db_path) or "."
        os.makedirs(self.db_folder, exist_ok=True)
        self.logger = logger
        self._initialize_database()

    def _initialize_database(self):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_updated
                ON chat_sessions(updated_at DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON chat_messages(session_id, timestamp)
            """)

            conn.commit()
            conn.close()

            self.logger.info(f"Database initialized successfully at {self.db_path}")

        except sqlite3.Error as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    def _get_connection(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise

    def create_session(self, session_id: str, title: str) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO chat_sessions (id, title) VALUES (?, ?)",
                (session_id, title[:50]),
            )

            conn.commit()
            conn.close()

            self.logger.info(f"Created session: {session_id} - '{title}'")
            return True

        except sqlite3.IntegrityError:
            self.logger.warning(f"Session {session_id} already exists")
            return False
        except sqlite3.Error as e:
            self.logger.error(f"Error creating session: {e}")
            return False

    def update_session_timestamp(self, session_id: str) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )

            conn.commit()
            conn.close()
            return True

        except sqlite3.Error as e:
            self.logger.error(f"Error updating session timestamp: {e}")
            return False

    def get_all_sessions(self) -> list[dict[str, Any]]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, title, created_at, updated_at
                FROM chat_sessions
                ORDER BY updated_at DESC
            """)

            rows = cursor.fetchall()
            conn.close()

            sessions = [dict(row) for row in rows]
            self.logger.info(f"Retrieved {len(sessions)} sessions")

            return sessions

        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving sessions: {e}")
            return []

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM chat_sessions
                WHERE id = ?
                """,
                (session_id,),
            )

            row = cursor.fetchone()
            conn.close()

            return dict(row) if row else None

        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving session: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))

            conn.commit()
            conn.close()

            self.logger.info(f"Deleted session: {session_id}")
            return True

        except sqlite3.Error as e:
            self.logger.error(f"Error deleting session: {e}")
            return False

    def save_message(self, session_id: str, role: str, content: str) -> int:
        if role not in ("user", "assistant"):
            raise ValueError("Role must be 'user' or 'assistant'")

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )

            message_id = cursor.lastrowid
            conn.commit()
            conn.close()

            self.update_session_timestamp(session_id)

            self.logger.info(f"Saved {role} message to session {session_id}")
            return message_id

        except sqlite3.Error as e:
            self.logger.error(f"Error saving message: {e}")
            return -1

    def get_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, session_id, role, content, timestamp
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
                """,
                (session_id,),
            )

            rows = cursor.fetchall()
            conn.close()

            messages = [dict(row) for row in rows]
            self.logger.info(f"Retrieved {len(messages)} messages for session {session_id}")

            return messages

        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving messages: {e}")
            return []
