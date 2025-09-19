import { getFunctions, httpsCallable, HttpsCallableResult } from "firebase/functions";
import { app } from './firebase';

// Defines the structure for session data
export interface SessionData {
  shipType?: string;
  departureDate?: string;
  startPort?: [number, number] | null;
  endPort?: [number, number] | null;
  optimizedRoute?: [number, number][];
  distance?: number;
  numSteps?: number;
  avgStepDistance?: number;
  shipDimensions?: {
    length: number;
    width: number;
    draft: number;
  };
}

// In-memory cache for the current session's data to reduce API calls
let tempSessionData: SessionData = {};

/**
 * Creates a new session on the backend or retrieves the existing one from localStorage.
 * @returns {Promise<string>} The session ID.
 */
export async function createOrGetSession(): Promise<string> {
  let sessionId = localStorage.getItem('sessionId');
  if (!sessionId) {
    const functions = getFunctions(app);
    const createSession = httpsCallable<void, { session_id: string }>(functions, 'create_session');
    try {
        const result: HttpsCallableResult<{ session_id: string }> = await createSession();
        sessionId = result.data.session_id;
        if (sessionId) {
          localStorage.setItem('sessionId', sessionId);
        } else {
          throw new Error("Failed to create session: No session_id returned");
        }
      } catch (error: unknown) {
        // Narrow the unknown before using it to avoid `any`
        if (error instanceof Error) {
          console.error("Error creating session:", error.message);
          throw error;
        }
        console.error("Error creating session:", String(error));
        throw new Error(String(error));
      }
  }
  return sessionId;
}

/**
 * Saves session data to the backend and updates the in-memory cache.
 * @param {string} sessionId - The ID of the session.
 * @param {Partial<SessionData>} data - The data to save.
 */
export async function saveSessionData(sessionId: string, data: Partial<SessionData>): Promise<void> {
  const functions = getFunctions(app);
  const updateSession = httpsCallable<{ session_id: string } & Partial<SessionData>, void>(functions, 'update_session');
  try {
    await updateSession({ session_id: sessionId, ...data });
    // Update in-memory cache
    tempSessionData = { ...tempSessionData, ...data };
  } catch (error: unknown) {
    if (error instanceof Error) {
      console.error("Error saving session data:", error.message);
      throw error;
    }
    console.error("Error saving session data:", String(error));
    throw new Error(String(error));
  }
}

/**
 * Retrieves session data, checking the in-memory cache first.
 * @param {string} sessionId - The ID of the session.
 * @returns {Promise<SessionData>} The session data.
 */
export async function getSessionData(sessionId: string): Promise<SessionData> {
  // First, check in-memory cache to avoid unnecessary network request
  if (Object.keys(tempSessionData).length > 0) {
    return tempSessionData;
  }

  const functions = getFunctions(app);
  const getSession = httpsCallable<{ session_id: string }, SessionData>(functions, 'get_session');
  try {
    const result: HttpsCallableResult<SessionData> = await getSession({ session_id: sessionId });
    // Update in-memory cache with fetched data
    tempSessionData = result.data;
    return result.data;
  } catch (error: unknown) {
    if (error instanceof Error) {
      console.error("Error getting session data:", error.message);
      throw error;
    }
    console.error("Error getting session data:", String(error));
    throw new Error(String(error));
  }
}

/**
 * Deletes a session from the backend and clears local storage.
 * @param {string} sessionId - The ID of the session to delete.
 */
export async function deleteSession(sessionId: string): Promise<void> {
  const functions = getFunctions(app);
  const deleteSessionFunc = httpsCallable<{ session_id: string }, void>(functions, 'delete_session');
  try {
    await deleteSessionFunc({ session_id: sessionId });
    localStorage.removeItem('sessionId');
    // Clear in-memory cache
    tempSessionData = {};
  } catch (error: unknown) {
    if (error instanceof Error) {
      console.error("Error deleting session:", error.message);
      throw error;
    }
    console.error("Error deleting session:", String(error));
    throw new Error(String(error));
  }
}

/**
 * Clears the temporary in-memory session data.
 */
export function clearTempSessionData(): void {
  tempSessionData = {};
}
