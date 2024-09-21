'use client'

import { useState } from 'react'
import Link from 'next/link'
import React from 'react'
import Image from 'next/image'

const CustomButton = ({ children, href, className, ...props }) => (
  <Link href={href}>
    <button className={`inline-flex items-center justify-center rounded-full bg-slate-950 px-8 py-3 text-sm font-medium text-white backdrop-blur-3xl transition-all hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-50 ${className}`} {...props}>
      {children}
    </button>
  </Link>
)

export default function HomePage() {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center bg-black text-white overflow-hidden">
      {/* Background Images */}
      <div className="absolute inset-0 z-0">
        <Image
          src="https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?ixlib=rb-4.0.3&auto=format&fit=crop&w=3440&q=80"
          alt="Background"
          layout="fill"
          objectFit="cover"
          className="opacity-50"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black to-transparent"></div>
      </div>
      
      {/* Content */}
      <div className="relative z-10 text-center">
        <h1 className="text-6xl font-bold mb-8 font-['Montserrat']">VEHICLE DETECTION APP</h1>
        <CustomButton
          href="/upload"
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          className={`relative overflow-hidden p-[1px] ${
            isHovered ? 'animate-hue-rotation' : ''
          }`}
        >
          <span className="absolute inset-[-1000%] animate-[spin_2s_linear_infinite] bg-[conic-gradient(from_90deg_at_50%_50%,#E2CBFF_0%,#393BB2_50%,#E2CBFF_100%)]" />
          <span className="relative z-10">START</span>
        </CustomButton>
      </div>
    </div>
  )
}