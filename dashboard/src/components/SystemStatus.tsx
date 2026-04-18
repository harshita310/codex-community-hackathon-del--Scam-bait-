import { motion } from 'framer-motion';
import { Cpu, Server, Database, Activity, Wifi } from 'lucide-react';

const StatusCard = ({ icon: Icon, label, value, subtext, color = 'mint' }: any) => (
  <div className="bg-white rounded-[24px] p-6 shadow-soft border border-white/60">
     <div className="flex justify-between items-start mb-4">
        <div className={`p-3 rounded-2xl ${color === 'mint' ? 'bg-mint-50 text-mint-600' : color === 'blue' ? 'bg-blue-50 text-blue-600' : 'bg-purple-50 text-purple-600'}`}>
            <Icon size={24} />
        </div>
        <div className="flex items-center gap-1 text-xs font-bold px-2 py-1 bg-gray-50 rounded-lg text-green-600">
            <Activity size={12} />
            Healthy
        </div>
     </div>
     <h3 className="text-2xl font-bold text-charcoal mb-1">{value}</h3>
     <p className="text-sm font-medium text-gray-400">{label}</p>
     <p className="text-xs text-gray-300 mt-4 pt-4 border-t border-gray-50">{subtext}</p>
  </div>
);

export const SystemStatus = () => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-6xl mx-auto"
    >
      <div className="mb-8">
           <h2 className="text-3xl font-bold text-charcoal">System Health</h2>
           <p className="text-gray-400 mt-1">Infrastructure performance monitoring</p>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
         <StatusCard icon={Cpu} label="CPU Usage" value="12%" subtext="4 Cores Active" color="mint" />
         <StatusCard icon={Database} label="Memory" value="2.4 GB" subtext="16 GB Total" color="blue" />
         <StatusCard icon={Wifi} label="Network" value="45 ms" subtext="Latency (avg)" color="purple" />
         <StatusCard icon={Server} label="Uptime" value="14d 2h" subtext="Since last reboot" color="mint" />
      </div>

      {/* Logs / Terminal View */}
      <div className="bg-[#1e1e1e] rounded-[32px] p-8 shadow-lg font-mono text-sm relative overflow-hidden">
         <div className="flex items-center gap-2 mb-6 border-b border-gray-700 pb-4">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="ml-4 text-gray-500">system_logs.log</span>
         </div>
         
         <div className="space-y-2 h-64 overflow-y-auto pr-4 text-gray-300">
            <div className="flex gap-4"><span className="text-gray-500">[21:30:05]</span> <span className="text-blue-400">INFO</span> <span>System initialized successfully.</span></div>
            <div className="flex gap-4"><span className="text-gray-500">[21:30:06]</span> <span className="text-blue-400">INFO</span> <span>Connected to database (SQLite optimized).</span></div>
            <div className="flex gap-4"><span className="text-gray-500">[21:30:12]</span> <span className="text-mint-400">SUCCESS</span> <span>Loaded 3 agent personas.</span></div>
            <div className="flex gap-4"><span className="text-gray-500">[21:32:45]</span> <span className="text-yellow-400">WARN</span> <span>High latency detected on node-3.</span></div>
            <div className="flex gap-4"><span className="text-gray-500">[21:35:00]</span> <span className="text-blue-400">INFO</span> <span>Cleaning up old session locks...</span></div>
            <div className="flex gap-4"><span className="text-gray-500">[21:38:22]</span> <span className="text-mint-400">SUCCESS</span> <span>Incoming connection accepted: Session #1042</span></div>
            <div className="flex gap-4"><span className="text-gray-500">[21:38:23]</span> <span className="text-purple-400">DEBUG</span> <span>Routing to 'Confused Grandfather' agent.</span></div>
            <div className="flex gap-4"><span className="text-gray-500">[21:40:01]</span> <span className="text-blue-400">INFO</span> <span>Heartbeat signal sent.</span></div>
            <div className="flex gap-4 animate-pulse"><span className="text-gray-500">[21:41:15]</span> <span className="text-blue-400">INFO</span> <span>Monitoring active...</span></div>
         </div>
      </div>
    </motion.div>
  );
};
