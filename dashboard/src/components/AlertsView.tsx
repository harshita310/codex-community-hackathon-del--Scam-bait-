import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Clock, XCircle, Search, Filter } from 'lucide-react';

const MOCK_ALERTS = [
  { id: 1, type: 'SCAM_DETECTED', source: '+91 98765 43210', confidence: 98, time: '2 mins ago', status: 'blocked', method: 'UPI Fraud' },
  { id: 2, type: 'SUSPICIOUS', source: '+91 88888 77777', confidence: 64, time: '15 mins ago', status: 'monitoring', method: 'Tech Support' },
  { id: 3, type: 'SCAM_DETECTED', source: '+91 74185 29630', confidence: 92, time: '1 hour ago', status: 'blocked', method: 'FedEx Scam' },
  { id: 4, type: 'INTEL_EXTRACTED', source: '+91 96385 27410', confidence: 100, time: '3 hours ago', status: 'logged', method: 'Data Harvesting' },
  { id: 5, type: 'SCAM_DETECTED', source: '+91 85274 19630', confidence: 88, time: '5 hours ago', status: 'blocked', method: 'Credit Card KY' },
];

export const AlertsView = () => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-5xl mx-auto h-[calc(100vh-140px)] flex flex-col"
    >
      <div className="flex justify-between items-end mb-8">
        <div>
           <h2 className="text-3xl font-bold text-charcoal">Threat Alerts</h2>
           <p className="text-gray-400 mt-1">Real-time log of detected security incidents</p>
        </div>
        
        <div className="flex gap-3">
            <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
                <input 
                    type="text" 
                    placeholder="Search logs..." 
                    className="pl-10 pr-4 py-2 bg-white rounded-xl border border-gray-100 shadow-sm focus:outline-none focus:ring-2 focus:ring-mint-100 w-64 text-sm"
                />
            </div>
            <button className="p-2 bg-white rounded-xl border border-gray-100 shadow-sm hover:bg-gray-50 text-gray-500">
                <Filter size={20} />
            </button>
        </div>
      </div>

      <div className="bg-white rounded-[32px] shadow-soft border border-white/60 flex-1 overflow-hidden flex flex-col">
          {/* Table Header */}
          <div className="grid grid-cols-12 gap-4 p-6 border-b border-gray-100 text-xs font-bold text-gray-400 uppercase tracking-wider bg-gray-50/50">
              <div className="col-span-1">Status</div>
              <div className="col-span-3">Source</div>
              <div className="col-span-3">Detection Type</div>
              <div className="col-span-2">Confidence</div>
              <div className="col-span-2">Time</div>
              <div className="col-span-1 text-right">Action</div>
          </div>

          {/* Table Body */}
          <div className="overflow-y-auto flex-1 p-2 space-y-2">
             {MOCK_ALERTS.map((alert) => (
                 <motion.div 
                    key={alert.id}
                    whileHover={{ scale: 1.005, backgroundColor: 'rgba(249, 250, 251, 0.8)' }}
                    className="grid grid-cols-12 gap-4 p-4 items-center rounded-2xl transition-colors cursor-pointer group"
                 >
                    <div className="col-span-1">
                        {alert.status === 'blocked' ? (
                            <div className="w-8 h-8 rounded-full bg-red-50 text-red-500 flex items-center justify-center">
                                <XCircle size={16} />
                            </div>
                        ) : alert.status === 'logged' ? (
                            <div className="w-8 h-8 rounded-full bg-blue-50 text-blue-500 flex items-center justify-center">
                                <CheckCircle size={16} />
                            </div>
                        ) : (
                             <div className="w-8 h-8 rounded-full bg-yellow-50 text-yellow-500 flex items-center justify-center">
                                <AlertTriangle size={16} />
                            </div>
                        )}
                    </div>
                    
                    <div className="col-span-3 font-semibold text-charcoal">{alert.source}</div>
                    
                    <div className="col-span-3">
                        <span className="px-3 py-1 bg-gray-100 rounded-lg text-xs font-medium text-gray-600">
                            {alert.method}
                        </span>
                    </div>

                    <div className="col-span-2">
                         <div className="flex items-center gap-2">
                             <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                 <div 
                                    className={`h-full rounded-full ${alert.confidence > 90 ? 'bg-mint-500' : alert.confidence > 70 ? 'bg-yellow-400' : 'bg-gray-300'}`} 
                                    style={{ width: `${alert.confidence}%` }} 
                                />
                             </div>
                             <span className="text-xs font-bold text-gray-400">{alert.confidence}%</span>
                         </div>
                    </div>

                    <div className="col-span-2 flex items-center gap-2 text-gray-400 text-sm">
                        <Clock size={14} />
                        {alert.time}
                    </div>

                    <div className="col-span-1 text-right">
                        <button className="text-xs font-bold text-charcoal hover:bg-gray-100 px-3 py-2 rounded-lg transition-colors opacity-0 group-hover:opacity-100">
                            Details
                        </button>
                    </div>
                 </motion.div>
             ))}
          </div>
      </div>
    </motion.div>
  );
};
