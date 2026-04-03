import sqlite3
import json
import threading
import os
from typing import Dict, Optional, List
from loguru import logger

# Corrected Import: Ensure correct pathing depends on sys.path
try:
    from models.schemas import PipelineState
except ImportError:
    # Fallback if run from core/ or via direct sub-pkg import
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.schemas import PipelineState

class SQLitePipelineStore:
    """
    SQLite-based persistent store for pipeline states.
    Provides long-term memory because the data is saved in a file on disk.
    """
    def __init__(self, db_filename: str = "recruitment_pipelines.db"):
        # Make the path absolute relative to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.db_path = os.path.join(project_root, db_filename)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database and creates the necessary tables."""
        try:
            with self._lock:
                # Ensure the DB directory exists
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                # Create a table to store pipeline state JSONs
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS pipelines (
                        pipeline_id TEXT PRIMARY KEY,
                        state_json TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                conn.close()
                logger.info(f"[SQLite Store] Initialized DB at {self.db_path}")
        except Exception as e:
            logger.error(f"[SQLite Store] Initialization error: {e}")
            raise

    def create(self, pipeline: PipelineState) -> PipelineState:
        """Persists a new pipeline state to the database."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                # Use pydantic v2 dump (v1 fallback)
                try:
                    state_json = pipeline.model_dump_json()
                except AttributeError:
                    state_json = pipeline.json()
                
                cursor.execute(
                    "INSERT INTO pipelines (pipeline_id, state_json) VALUES (?, ?)",
                    (pipeline.pipeline_id, state_json)
                )
                conn.commit()
                conn.close()
                logger.info(f"[SQLite Store] Created pipeline {pipeline.pipeline_id}")
                return pipeline
        except Exception as e:
            logger.error(f"[SQLite Store] Create error: {e}")
            raise

    def get(self, pipeline_id: str) -> Optional[PipelineState]:
        """Retrieves a pipeline state by its ID."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute("SELECT state_json FROM pipelines WHERE pipeline_id = ?", (pipeline_id,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    # Reconstruct using pydantic v2 (v1 fallback)
                    try:
                        return PipelineState.model_validate_json(row[0])
                    except AttributeError:
                        return PipelineState.parse_raw(row[0])
                return None
        except Exception as e:
            logger.error(f"[SQLite Store] Get error: {e}")
            return None

    def update(self, pipeline: PipelineState) -> PipelineState:
        """Updates an existing pipeline state in the database."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                try:
                    state_json = pipeline.model_dump_json()
                except AttributeError:
                    state_json = pipeline.json()
                
                cursor.execute(
                    "UPDATE pipelines SET state_json = ?, updated_at = CURRENT_TIMESTAMP WHERE pipeline_id = ?",
                    (state_json, pipeline.pipeline_id)
                )
                conn.commit()
                conn.close()
                logger.debug(f"[SQLite Store] Updated pipeline {pipeline.pipeline_id}")
                return pipeline
        except Exception as e:
            logger.error(f"[SQLite Store] Update error: {e}")
            raise

    def list_all(self) -> List[PipelineState]:
        """Lists all persistent pipeline states."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute("SELECT state_json FROM pipelines")
                rows = cursor.fetchall()
                conn.close()
                
                pipelines = []
                for row in rows:
                    try:
                        p = PipelineState.model_validate_json(row[0])
                    except AttributeError:
                        p = PipelineState.parse_raw(row[0])
                    pipelines.append(p)
                return pipelines
        except Exception as e:
            logger.error(f"[SQLite Store] List error: {e}")
            return []

    def delete(self, pipeline_id: str) -> bool:
        """Deletes a pipeline's state from the database."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM pipelines WHERE pipeline_id = ?", (pipeline_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
                conn.close()
                return deleted
        except Exception as e:
            logger.error(f"[SQLite Store] Delete error: {e}")
            return False

    def __len__(self) -> int:
        """Returns the number of persisted pipelines."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM pipelines")
                count = cursor.fetchone()[0]
                conn.close()
                return count
        except Exception:
            return 0

    def __iter__(self):
        """Allows iteration over stored pipeline IDs for backward compatibility."""
        pipelines = self.list_all()
        for p in pipelines:
            yield p.pipeline_id

# Singleton store instance
# This provides persistent "long-term memory" across server restarts.
pipeline_store = SQLitePipelineStore()
