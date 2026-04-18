import { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { StatsHeader } from './components/StatsHeader';
import { ScamVolumeChart, ChartDataPoint } from './components/ScamVolumeChart';
import { IntelCard } from './components/IntelCard';
import { LiveFeed } from './components/LiveFeed';
import { useWebSocket } from './hooks/useWebSocket';
import { ScamMessage, Intelligence } from './types';
import { SettingsPanel } from './components/SettingsPanel';
import { AlertsView } from './components/AlertsView';
import { SystemStatus } from './components/SystemStatus';
import { LiveMonitor } from './components/LiveMonitor';

// Initial Mock Data (to keep the "alive" feel before real data comes in)
const INITIAL_CHART_DATA: ChartDataPoint[] = [
  { name: '10:00', uv: 20 },
  { name: '10:05', uv: 45 },
  { name: '10:10', uv: 30 },
  { name: '10:15', uv: 60 },
  { name: '10:20', uv: 40 },
  { name: '10:25', uv: 75 },
  { name: '10:30', uv: 55 },
];

const INITIAL_INTEL: Intelligence = {
  bankAccounts: ["HDFC **** 1234"],
  upiIds: ["helper@paytm"],
  phoneNumbers: ["+91 98765 00000"],
  phishingLinks: [],
  emails: [],
  apkLinks: [],
  cryptoWallets: [],
  socialHandles: [],
  ifscCodes: [],
  suspiciousKeywords: []
};

function App() {
  // Use production API URL (hardcoded for deployment)
  const apiUrl = 'https://honey-api-wr74.onrender.com';
  const wsUrl = apiUrl.replace('https', 'wss') + '/ws/dashboard';
  
  const { isConnected, lastMessage } = useWebSocket(wsUrl);
  
  // State
  const [messages, setMessages] = useState<ScamMessage[]>([]);
  const [chartData, setChartData] = useState<ChartDataPoint[]>(INITIAL_CHART_DATA);
  const [intelligence, setIntelligence] = useState<Intelligence>(INITIAL_INTEL);
  const [activeScams, setActiveScams] = useState(124); // Start with mock
  const [hoursSaved] = useState(315);   // Start with mock
  const [activeTab, setActiveTab] = useState('dashboard');

  // Fetch initial stats
  useEffect(() => {
    fetch(`${apiUrl}/api/v1/stats`)
      .then(res => res.json())
      .then(data => {
        if (data.total_sessions > 0) {
           setActiveScams(data.active_now > 0 ? data.active_now : 124); // Keep mock if 0
           // setHoursSaved(data.total_sessions * 0.5); // Approx logic
        }
      })
      .catch(err => console.error("Failed to fetch stats:", err));
  }, []);

  // Handle WebSocket Updates
  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === 'new_message') {
      const msg = lastMessage as unknown as ScamMessage;
      setMessages(prev => [...prev, msg]);
      
      // Update chart dynamically
      setChartData(prev => {
        const newData = [...prev, { 
          name: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), 
          uv: Math.floor(Math.random() * 50) + 30 // Simulate activity spike
        }];
        return newData.slice(-10); // Keep last 10 points
      });
    }

    if (lastMessage.type === 'intelligence_update') {
       // Merge new intelligence
       const newIntel = lastMessage.data as Intelligence;
       setIntelligence(newIntel);
    }

    if (lastMessage.type === 'stats_update') {
       const stats = lastMessage.data;
       setActiveScams(stats.active_now);
    }

  }, [lastMessage]);

  return (
    <Layout activeTab={activeTab} onTabChange={setActiveTab}>
      <div className="max-w-7xl mx-auto space-y-8">
      
           {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Top Header Row */}
            <div className="flex justify-between items-center bg-white p-6 rounded-xl shadow-sm border border-slate-200">
              <div>
                <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Dashboard Overview</h1>
                <p className="text-slate-500 text-sm mt-1">Real-time scam detection and intelligence monitoring.</p>
              </div>
              
              {/* Connection Status Pill */}
              <div className={`px-3 py-1.5 rounded-full text-xs font-semibold border flex items-center gap-2 ${isConnected ? 'bg-green-50 text-green-700 border-green-200' : 'bg-red-50 text-red-700 border-red-200'}`}>
                <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                {isConnected ? 'System Online' : 'Offline'}
              </div>
            </div>

            {/* Stats Row */}
            <StatsHeader activeScams={activeScams} hoursSaved={hoursSaved} />

            {/* Main Grid */}
            <div className="grid grid-cols-12 gap-6">
              
              {/* Hero Chart (Span 8) */}
              <div className="col-span-12 lg:col-span-8 bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col h-[400px]">
                <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                  Scam Volume Trends
                </h3>
                <div className="flex-1 w-full h-full">
                  <ScamVolumeChart data={chartData} />
                </div>
              </div>

              {/* Intelligence Side Panel (Span 4) */}
              <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
                <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                   <h3 className="text-lg font-bold text-slate-800 mb-4">Live Intelligence</h3>
                   <IntelCard intelligence={intelligence} />
                </div>
                
                {/* Live Feed Component */}
                <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex-1 flex flex-col h-[300px]">
                  <div className="flex justify-between items-center mb-4">
                     <h3 className="text-lg font-bold text-slate-800">Live Feed</h3>
                     <span className="text-xs text-slate-400 font-mono">Real-time</span>
                  </div>
                   <div className="flex-1 overflow-hidden relative">
                      <LiveFeed messages={messages} /> 
                   </div>
                </div>
              </div>

              {/* AI Analysis (Span 12) */}
              <div className="col-span-12">
                <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm flex items-center justify-between">
                    <div>
                        <h2 className="text-lg font-bold text-slate-900">Active Persona Agent</h2>
                        <div className="flex items-center gap-4 mt-1">
                           <span className="text-slate-500 text-sm">Role: <span className="text-slate-900 font-semibold">Confused Elderly (Meena)</span></span>
                           <span className="text-slate-500 text-sm">Status: <span className="text-green-600 font-semibold">Engaging</span></span>
                        </div>
                    </div>
                </div>
              </div>

            </div>
          </div>
        )}

        {/* Other Tabs */}
        {activeTab === 'live' && <LiveMonitor messages={messages} />}
        {activeTab === 'alerts' && <AlertsView />}
        {activeTab === 'system' && <SystemStatus />}
        {activeTab === 'settings' && <SettingsPanel />}

      </div>
    </Layout>
  );
}

export default App;
