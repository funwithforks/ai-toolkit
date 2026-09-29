'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Home, Settings, BrainCircuit, Images, Plus, X, Sparkles } from 'lucide-react';
import { FaXTwitter, FaDiscord, FaYoutube } from 'react-icons/fa6';
import { createGlobalState } from 'react-global-hooks';
import ThemeToggle from './ThemeToggle';
import ThemeLogo from './ThemeLogo';
import ActiveJobWidget from './ActiveJobWidget';
import OstrisCloudBalance from './OstrisCloudBalance';

export const mobileSidebarState = createGlobalState<boolean>(false);

const Sidebar = () => {
  const [isMobileOpen, setIsMobileOpen] = mobileSidebarState.use();
  const pathname = usePathname();

  // Close mobile menu on route change
  useEffect(() => {
    setIsMobileOpen(false);
  }, [pathname]);

  // Lock body scroll when mobile menu open
  useEffect(() => {
    if (isMobileOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isMobileOpen]);

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home },
    { name: 'Generate', href: '/generate', icon: Sparkles, badge: 'alpha' },
    { name: 'New Job', href: '/jobs/new', icon: Plus },
    { name: 'Queue', href: '/jobs', icon: BrainCircuit },
    { name: 'Datasets', href: '/datasets', icon: Images },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  const socialsBoxClass =
    'flex flex-col items-center justify-center p-1 hover:bg-gray-800 rounded-lg transition-colors';
  const socialIconClass = 'w-5 h-5 text-gray-400 hover:text-white';

  const sidebarContent = (
    <>
      <div className="px-4 py-3 flex items-center justify-between">
        <h1 className="text-l">
          <ThemeLogo />
          <span className="font-bold uppercase">Ostris</span>
          <span className="ml-2 uppercase text-gray-300">AI-Toolkit</span>
        </h1>
        <button
          onClick={() => setIsMobileOpen(false)}
          className="md:hidden text-gray-400 hover:text-white p-1"
          aria-label="Close menu"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
      <OstrisCloudBalance />
      <nav className="flex-1">
        <ul className="px-2 py-4 space-y-2">
          {navigation.map(item => (
            <li key={item.name}>
              <Link
                href={item.href}
                className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors"
              >
                <item.icon className="w-5 h-5 mr-3" />
                {item.name}
                {'badge' in item && item.badge && (
                  <span className="ml-2 px-1.5 py-0.5 text-[10px] leading-none uppercase tracking-wide rounded bg-blue-600/20 text-blue-400 border border-blue-500/40">
                    {item.badge}
                  </span>
                )}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
      <ActiveJobWidget />

      {/* Social links grid */}
      <div className="px-1 py-1 border-t border-gray-800">
        <div className="grid grid-cols-4 gap-4">
          <a href="https://discord.gg/VXmU2f5WEU" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaDiscord className={socialIconClass} />
          </a>
          <a href="https://www.youtube.com/@ostrisai" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaYoutube className={socialIconClass} />
          </a>
          <a href="https://x.com/ostrisai" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaXTwitter className={socialIconClass} />
          </a>
          <ThemeToggle />
        </div>
      </div>
      <div className="text-center text-[10px] text-gray-400 py-1 bg-gray-800">
        Ostris AI-Toolkit v{process.env.NEXT_PUBLIC_APP_VERSION}
      </div>
    </>
  );

  return (
    <>
      {/* Desktop sidebar - always visible on md+ */}
      <div className="hidden md:flex flex-col w-59 bg-gray-900 text-gray-100">{sidebarContent}</div>

      {/* Mobile overlay sidebar */}
      <div
        className={`md:hidden fixed inset-0 bg-black/60 z-40 transition-opacity duration-300 ease-in-out ${
          isMobileOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
        onClick={() => setIsMobileOpen(false)}
        aria-hidden="true"
      />
      <div
        className={`md:hidden fixed top-0 left-0 bottom-0 w-64 max-w-[85vw] bg-gray-900 text-gray-100 z-50 flex flex-col shadow-xl transform transition-transform duration-300 ease-in-out ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {sidebarContent}
      </div>
    </>
  );
};

export default Sidebar;
