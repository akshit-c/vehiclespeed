import React from 'react';
import './globals.css'
import { Inter, Montserrat } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })
const montserrat = Montserrat({ weight: '700', subsets: ['latin'] })

export const metadata = {
  title: 'Vehicle Detection App',
  description: 'Detect vehicles and recognize number plates',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} ${montserrat.className}`}>{children}</body>
    </html>
  )
}