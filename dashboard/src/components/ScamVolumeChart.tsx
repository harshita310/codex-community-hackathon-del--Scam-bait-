import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';
import { motion } from 'framer-motion';

export interface ChartDataPoint {
  name: string;
  uv: number;
}

interface ScamVolumeChartProps {
  data: ChartDataPoint[];
}

export const ScamVolumeChart = ({ data }: ScamVolumeChartProps) => {
  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="w-full h-full relative min-h-[300px]"
    >
      <div className="absolute top-0 left-2 z-10">
        <p className="text-xs font-mono text-gray-400">LIVE TRAFFIC • LAST 24 HOURS</p>
      </div>

      <div className="h-full w-full -ml-4 mt-4">
        <ResponsiveContainer width="105%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorUv" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <YAxis hide />
            <Area 
              type="monotone" 
              dataKey="uv" 
              stroke="#3b82f6" 
              strokeWidth={2}
              fillOpacity={1} 
              fill="url(#colorUv)" 
              isAnimationActive={true}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
};
