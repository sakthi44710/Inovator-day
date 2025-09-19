'use client';

import { motion } from 'framer-motion'
import RouteForm from './RouteForm'
import { ChevronLeft, ChevronRight } from 'lucide-react'

interface SidebarProps {
  isNavOpen: boolean
  setIsNavOpen: (isOpen: boolean) => void
  setSelectedRoute: (route: [number, number][]) => void
  startPort: [number, number] | null
  endPort: [number, number] | null
  setIsSelectingLocation: (type: 'start' | 'end' | null) => void
}

export default function Sidebar({ isNavOpen, setIsNavOpen, setSelectedRoute, startPort, endPort, setIsSelectingLocation }: SidebarProps) {
  return (
    <motion.div 
      className={`bg-white dark:bg-gray-800 shadow-lg overflow-y-auto transition-all duration-300 ease-in-out relative flex flex-col`}
      initial={false}
      animate={{ width: isNavOpen ? '24rem' : '5rem' }}
    >
      <div className="p-6 flex-grow">
        <motion.h1 
          className="text-3xl font-bold mb-6 text-emerald-600 dark:text-emerald-400 whitespace-nowrap overflow-hidden"
          animate={{ opacity: isNavOpen ? 1 : 0, transition: { delay: 0.1 } }}
        >
          {isNavOpen ? 'Ship Route Optimizer' : 'SRO'}
        </motion.h1>
        <div className={!isNavOpen ? 'hidden' : ''}>
            <RouteForm
              setSelectedRoute={setSelectedRoute}
              isNavOpen={isNavOpen}
              startPort={startPort}
              endPort={endPort}
              setIsSelectingLocation={setIsSelectingLocation}
            />
        </div>
      </div>
      <div className="p-4 flex justify-end">
        <button
          onClick={() => setIsNavOpen(!isNavOpen)}
          className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 p-2 rounded-full shadow-md z-10 hover:bg-gray-300 dark:hover:bg-gray-600"
        >
          {isNavOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
        </button>
      </div>
    </motion.div>
  )
}
