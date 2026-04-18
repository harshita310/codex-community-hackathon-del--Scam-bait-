import { ShieldCheck, Timer } from 'lucide-react';

interface StatsHeaderProps {
  activeScams: number;
  hoursSaved: number;
}

export const StatsHeader = ({ activeScams, hoursSaved }: StatsHeaderProps) => {
  return (
    <div className="flex gap-4 mb-6">
      <div className="flex-1 bg-white rounded-xl p-6 flex items-center gap-6 shadow-sm border border-slate-200">
        <div className="p-4 bg-blue-50 rounded-xl text-blue-600">
          <ShieldCheck size={28} />
        </div>
        <div>
          <h2 className="text-3xl font-bold text-slate-900">{activeScams}</h2>
          <p className="text-sm font-medium text-slate-500">Scams Intercepted</p>
        </div>
      </div>

      <div className="flex-1 bg-white rounded-xl p-6 flex items-center gap-6 shadow-sm border border-slate-200">
        <div className="p-4 bg-purple-50 rounded-xl text-purple-600">
          <Timer size={28} />
        </div>
        <div>
          <h2 className="text-3xl font-bold text-slate-900">{hoursSaved}</h2>
          <p className="text-sm font-medium text-slate-500">Hours Wasted</p>
        </div>
      </div>
      
       {/* Small circular usage stat */}
       <div className="bg-white rounded-xl p-4 flex flex-col items-center justify-center w-32 shadow-sm border border-slate-200">
          <div className="text-xl font-bold text-slate-900">84%</div>
          <div className="text-[10px] text-slate-400 uppercase font-semibold">Load</div>
       </div>
    </div>
  );
};
