export interface ScamMessage {
  webhookId: string;
  sessionId: string;
  sender: string;
  text: string;
  timestamp: string;
}

export type ScamType = 
  | "DIGITAL_ARREST" 
  | "UPI_SCAM" 
  | "JOB_SCAM" 
  | "SEXTORTION" 
  | "LOTTERY_SCAM" 
  | "NONE" 
  | "UNKNOWN";

export interface Intelligence {
  bankAccounts: string[];
  upiIds: string[];
  phishingLinks: string[];
  phoneNumbers: string[];
  emails: string[];
  apkLinks: string[];
  cryptoWallets: string[];
  socialHandles: string[];
  ifscCodes: string[];
  suspiciousKeywords: string[];
}

export interface DashboardStats {
  total_sessions: number;
  active_now: number;
  intelligence_items: number;
  scams_detected: number;
}

export interface WebSocketEvent {
  type: "new_message" | "intelligence_update" | "scam_detected" | "stats_update";
  data: any;
  timestamp?: string;
}
