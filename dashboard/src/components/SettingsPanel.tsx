import { useState } from 'react';
import { motion } from 'framer-motion';
import { Bell, Shield, User, Volume2, Lock, Save } from 'lucide-react';

export const SettingsPanel = () => {
  const [systemActive, setSystemActive] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [autoBlock, setAutoBlock] = useState(true);
  const [persona, setPersona] = useState('grandfather');
  const [sensitivity, setSensitivity] = useState(75);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl mx-auto space-y-8 pb-12"
    >
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-charcoal">Settings</h2>
          <p className="text-gray-400 mt-1">Configure your active defense parameters</p>
        </div>
        <button className="flex items-center gap-2 bg-charcoal text-white px-6 py-3 rounded-2xl hover:bg-black transition-colors shadow-lg shadow-charcoal/20">
          <Save size={18} />
          <span className="font-semibold text-sm">Save Changes</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* System Control Card */}
        <div className="bg-white rounded-[32px] p-8 shadow-soft border border-white/60 col-span-2">
            <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-mint-50 text-mint-600 rounded-full">
                        <Shield size={24} />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-charcoal">System Status</h3>
                        <p className="text-gray-400 text-sm">Main kill-switch for the honeypot</p>
                    </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" checked={systemActive} onChange={() => setSystemActive(!systemActive)} className="sr-only peer" />
                    <div className="w-14 h-8 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-mint-100 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-7 after:w-7 after:transition-all peer-checked:bg-mint-500"></div>
                </label>
            </div>
            <div className="p-4 bg-gray-50 rounded-2xl border border-dashed border-gray-200">
                <p className="text-sm text-gray-500 leading-relaxed">
                    <span className="font-bold text-charcoal">Current Mode: </span> 
                    {systemActive ? "Active Defense (Routing calls to AI)" : "Passive Monitoring (Logging only)"}
                </p>
            </div>
        </div>

        {/* Persona Selection */}
        <div className="bg-white rounded-[32px] p-8 shadow-soft border border-white/60">
            <div className="flex items-center gap-4 mb-6">
                <div className="p-3 bg-purple-50 text-purple-600 rounded-full">
                    <User size={24} />
                </div>
                <h3 className="text-xl font-bold text-charcoal">Active Persona</h3>
            </div>
            
            <div className="space-y-3">
                {['grandfather', 'student', 'executive'].map((p) => (
                    <div 
                        key={p} 
                        onClick={() => setPersona(p)}
                        className={`p-4 rounded-xl border cursor-pointer transition-all ${
                            persona === p 
                            ? 'border-purple-500 bg-purple-50/50' 
                            : 'border-gray-100 hover:border-purple-200'
                        }`}
                    >
                        <div className="flex items-center justify-between">
                            <span className="font-semibold text-charcoal capitalize">
                                {p === 'grandfather' ? 'Confused Grandfather' : 
                                 p === 'student' ? 'Tech-Savvy Student' : 'Busy Executive'}
                            </span>
                            {persona === p && <div className="w-3 h-3 bg-purple-500 rounded-full shadow-sm" />}
                        </div>
                    </div>
                ))}
            </div>
        </div>

        {/* Sensitivity & Notifications */}
        <div className="bg-white rounded-[32px] p-8 shadow-soft border border-white/60 space-y-8">
            {/* Sensitivity Slider */}
            <div>
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                        <Volume2 size={20} className="text-gray-400" />
                        <span className="font-bold text-charcoal">Detection Sensitivity</span>
                    </div>
                    <span className="bg-gray-100 px-2 py-1 rounded-lg text-xs font-semibold">{sensitivity}%</span>
                </div>
                <input 
                    type="range" 
                    min="0" 
                    max="100" 
                    value={sensitivity} 
                    onChange={(e) => setSensitivity(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-100 rounded-lg appearance-none cursor-pointer accent-charcoal"
                />
                <p className="text-xs text-gray-400 mt-2">Higher sensitivity may increase false positives.</p>
            </div>

            {/* Notification Toggles */}
            <div className="space-y-4 pt-4 border-t border-gray-100">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Bell size={20} className="text-gray-400" />
                        <span className="font-medium text-charcoal">Push Notifications</span>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" checked={notifications} onChange={() => setNotifications(!notifications)} className="sr-only peer" />
                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-charcoal"></div>
                    </label>
                </div>

                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Lock size={20} className="text-gray-400" />
                        <span className="font-medium text-charcoal">Auto-Block Scammers</span>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" checked={autoBlock} onChange={() => setAutoBlock(!autoBlock)} className="sr-only peer" />
                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-charcoal"></div>
                    </label>
                </div>
            </div>
        </div>

      </div>
    </motion.div>
  );
};
