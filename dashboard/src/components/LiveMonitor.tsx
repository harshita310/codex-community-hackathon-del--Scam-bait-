import { motion } from 'framer-motion';
import { PhoneCall, Mic, MoreVertical, StopCircle } from 'lucide-react';
import { LiveFeed } from './LiveFeed';
import { ScamMessage } from '../types';

interface LiveMonitorProps {
  messages: ScamMessage[];
}

export const LiveMonitor = ({ messages }: LiveMonitorProps) => {
  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="h-[calc(100vh-100px)] flex gap-6"
    >
      {/* Left Column: Active Call Visualization */}
      <div className="flex-1 flex flex-col gap-6">
         {/* Live Status Card */}
         <div className="bg-gradient-to-br from-charcoal to-black rounded-[32px] p-8 text-white shadow-lg flex-1 relative overflow-hidden flex flex-col items-center justify-center">
             
             {/* Dynamic Waveform Animation (Mock) */}
             <div className="flex gap-1 h-32 items-center mb-8">
                 {[...Array(20)].map((_, i) => (
                     <motion.div 
                        key={i}
                        animate={{ height: [20, Math.random() * 80 + 20, 20] }}
                        transition={{ repeat: Infinity, duration: 1, delay: i * 0.05 }}
                        className="w-2 bg-mint-500 rounded-full opacity-60"
                     />
                 ))}
             </div>

             <div className="text-center z-10">
                 <div className="w-24 h-24 bg-white/10 rounded-full mx-auto mb-6 flex items-center justify-center backdrop-blur-md border border-white/10 relative">
                    <PhoneCall size={32} className="text-white" />
                    <div className="absolute inset-0 rounded-full border border-mint-500 animate-ping opacity-20" />
                 </div>
                 <h2 className="text-3xl font-bold mb-2">+91 98765 00000</h2>
                 <p className="text-gray-400">Active Call • 04:23</p>
             </div>

             {/* Action Buttons */}
             <div className="flex gap-4 mt-12">
                 <button className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center hover:bg-white/20 transition-colors">
                     <Mic size={20} />
                 </button>
                 <button className="w-16 h-16 rounded-full bg-red-500 flex items-center justify-center hover:bg-red-600 transition-colors shadow-lg shadow-red-500/30">
                     <StopCircle size={28} fill="white" />
                 </button>
                 <button className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center hover:bg-white/20 transition-colors">
                     <MoreVertical size={20} />
                 </button>
             </div>
             
             {/* Background glow */}
             <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-transparent to-black/50 pointer-events-none" />
         </div>
      </div>

      {/* Right Column: Transcript & Analysis */}
      <div className="w-[450px] flex flex-col gap-6">
          <div className="bg-white rounded-[32px] p-6 shadow-soft h-[60%] border border-white/60 relative overflow-hidden">
               <h3 className="text-lg font-bold text-charcoal mb-4 px-2">Live Transcript</h3>
               <div className="h-[calc(100%-40px)]">
                  <LiveFeed messages={messages} />
               </div>
          </div>

          <div className="bg-white rounded-[32px] p-6 shadow-soft flex-1 border border-white/60">
              <h3 className="text-lg font-bold text-charcoal mb-4">Real-Time Analysis</h3>
              <div className="space-y-4">
                  <div>
                      <div className="flex justify-between text-xs font-semibold text-gray-400 mb-1">
                          <span>SCAM PROBABILITY</span>
                          <span className="text-red-500">HIGH</span>
                      </div>
                      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                          <div className="h-full bg-red-500 w-[92%] rounded-full" />
                      </div>
                  </div>
                  
                  <div className="p-4 bg-gray-50 rounded-2xl text-sm text-gray-600 border border-gray-100">
                      <span className="font-bold text-charcoal block mb-1">Detected Intent:</span>
                      Scammer is attempting to create urgency regarding a "blocked parcel". Standard FedEx fraud script detected.
                  </div>
              </div>
          </div>
      </div>
    </motion.div>
  );
};
