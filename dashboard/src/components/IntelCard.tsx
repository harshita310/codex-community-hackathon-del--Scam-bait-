import { motion } from 'framer-motion';
import { Phone, Building2, Link2, CreditCard, MapPin, AtSign } from 'lucide-react';
import { Intelligence } from '../types';

interface IntelCardProps {
  intelligence: Intelligence;
}

export const IntelCard = ({ intelligence }: IntelCardProps) => {
  const allItems = [
    ...intelligence.phoneNumbers.map(v => ({ type: 'Phone', value: v })),
    ...intelligence.bankAccounts.map(v => ({ type: 'Bank', value: v })),
    ...intelligence.upiIds.map(v => ({ type: 'UPI', value: v })),
    ...intelligence.emails.map(v => ({ type: 'Email', value: v })),
    ...intelligence.cryptoWallets.map(v => ({ type: 'Crypto', value: v })),
    ...intelligence.phishingLinks.map(v => ({ type: 'Link', value: v })),
  ];

  return (
    <div className="h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        {/* Header removed as it is in parent */}
      </div>
      
      <div className="flex flex-col gap-2 overflow-y-auto pr-2 custom-scrollbar flex-1">
        {allItems.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-slate-400 gap-2">
            <div className="w-8 h-8 rounded-full border-2 border-dashed border-slate-300 animate-spin" />
            <span className="text-xs">Waiting for data...</span>
          </div>
        ) : (
          allItems.map((item, idx) => (
            <motion.div
              key={item.value + idx}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg border border-slate-100 hover:bg-slate-100 transition-colors"
            >
              <div className="text-blue-500 bg-blue-50 p-1.5 rounded-md">
                {getIcon(item.type)}
              </div>
              <span className="text-sm font-medium text-slate-700 font-mono">{item.value}</span>
              <span className="ml-auto text-[10px] uppercase font-bold text-slate-400 bg-white px-1.5 py-0.5 rounded border border-slate-200">
                {item.type}
              </span>
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
};

const getIcon = (type: string) => {
  switch (type) {
    case 'Phone': return <Phone size={12} />;
    case 'Bank': return <Building2 size={12} />;
    case 'UPI': return <CreditCard size={12} />;
    case 'Email': return <AtSign size={12} />;
    case 'Link': return <Link2 size={12} />;
    default: return <MapPin size={12} />;
  }
};
