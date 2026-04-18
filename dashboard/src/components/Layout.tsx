import React from 'react';
import { LayoutDashboard, Radio, Settings, ShieldAlert, Cpu } from 'lucide-react';

interface SidebarItemProps {
  icon: React.ElementType;
  active?: boolean;
}

interface LayoutProps {
  children: React.ReactNode;
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export const Layout = ({ children, activeTab, onTabChange }: LayoutProps) => {
  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-slate-200 flex flex-col z-10 shadow-sm">
        <div className="p-6 border-b border-slate-100 flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white">
            <ShieldAlert size={18} />
          </div>
          <h1 className="font-bold text-slate-800 tracking-tight">ScamBait AI</h1>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          <SidebarItem label="Dashboard" icon={LayoutDashboard} active={activeTab === 'dashboard'} onClick={() => onTabChange('dashboard')} />
          <SidebarItem label="Live Feed" icon={Radio} active={activeTab === 'live'} onClick={() => onTabChange('live')} />
          <SidebarItem label="Intelligence" icon={ShieldAlert} active={activeTab === 'alerts'} onClick={() => onTabChange('alerts')} />
          <SidebarItem label="System" icon={Cpu} active={activeTab === 'system'} onClick={() => onTabChange('system')} />
        </nav>

        <div className="p-4 border-t border-slate-100">
          <SidebarItem label="Settings" icon={Settings} active={activeTab === 'settings'} onClick={() => onTabChange('settings')} />
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto bg-slate-50 relative">
        <div className="max-w-7xl mx-auto p-8">
           {children}
        </div>
      </main>
    </div>
  );
};

interface SidebarItemPropsExtended extends SidebarItemProps {
  label: string;
  onClick: () => void;
}

const SidebarItem = ({ icon: Icon, label, active, onClick }: SidebarItemPropsExtended) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
      active 
        ? 'bg-blue-50 text-blue-700' 
        : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
    }`}
  >
    <Icon size={18} />
    {label}
  </button>
);
