import { useState, useEffect, useRef, useCallback } from 'react';
import { WebSocketEvent } from '../types';

export function useWebSocket(url: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketEvent | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<number>();

  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        console.log('✅ WebSocket Connected');
        setIsConnected(true);
        // Send initial ping to keep alive if needed
        ws.current?.send("ping");
      };

      ws.current.onmessage = (event) => {
        if (event.data === "pong") return;
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch (e) {
          console.error('Failed to parse WS message:', e);
        }
      };

      ws.current.onclose = () => {
        console.log('❌ WebSocket Disconnected');
        setIsConnected(false);
        // Reconnect after 3 seconds
        reconnectTimeout.current = setTimeout(connect, 3000);
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket Error:', error);
        ws.current?.close();
      };

    } catch (error) {
      console.error('Connection failed:', error);
      reconnectTimeout.current = setTimeout(connect, 3000);
    }
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      if (ws.current) {
        ws.current.close();
      }
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
    };
  }, [connect]);

  return { isConnected, lastMessage };
}
