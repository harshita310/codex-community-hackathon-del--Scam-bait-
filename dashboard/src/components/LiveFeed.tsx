import { AnimatePresence, motion } from 'framer-motion';
import { ShieldAlert, User } from 'lucide-react';
import { cn, formatTime } from '../lib/utils';
import { ScamMessage } from '../types';

interface LiveFeedProps {
  messages: ScamMessage[];
}

export const LiveFeed = ({ messages }: LiveFeedProps) => {
  const displayMessages = messages.slice(-10); // Show last 10

  return (
    <div className="h-full w-full overflow-y-auto space-y-3 custom-scrollbar pr-2 pb-2">
      <AnimatePresence initial={false}>
        {displayMessages.map((msg, idx) => {
          const isScammer = msg.sender === 'scammer';
          return (
            <motion.div
              key={msg.timestamp + idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={cn(
                "flex gap-3 p-3 rounded-xl text-sm max-w-[90%]",
                isScammer 
                  ? "bg-red-50 text-red-900 self-start mr-auto border border-red-100 shadow-sm" 
                  : "bg-blue-50 text-blue-900 self-end ml-auto border border-blue-100 shadow-sm"
              )}
            >
              <div className={cn(
                "w-8 h-8 rounded-full flex items-center justify-center shrink-0 border",
                isScammer ? "bg-white border-red-200 text-red-500" : "bg-white border-blue-200 text-blue-500"
              )}>
                {isScammer ? <ShieldAlert size={14} /> : <User size={14} />}
              </div>
              <div>
                 <p className="font-semibold text-xs mb-1 opacity-70">
                   {isScammer ? "Scammer" : "Meena (Bot)"} • {formatTime(msg.timestamp)}
                 </p>
                 <p className="leading-relaxed text-slate-700">{msg.text}</p>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
      {displayMessages.length === 0 && (
         <div className="h-full flex flex-col items-center justify-center text-slate-400 gap-2">
            <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center">
                <ShieldAlert className="text-slate-300" />
            </div>
            <span className="text-sm font-medium">Waiting for traffic...</span>
         </div>
      )}
    </div>
  );
};
