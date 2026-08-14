import { NextRequest, NextResponse } from 'next/server';
import { getHistoricalData } from '@/lib/forecastEngine';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  try {
    const data = getHistoricalData();
    return NextResponse.json({
      success: true,
      total_count: data.length,
      data
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error?.message || 'Failed to fetch historical dataset' },
      { status: 500 }
    );
  }
}
