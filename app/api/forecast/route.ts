import { NextRequest, NextResponse } from 'next/server';
import { runForecastSimulation } from '@/lib/forecastEngine';
import { ForecastRequest } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  try {
    const body: Partial<ForecastRequest> = await req.json();

    const requestParams: ForecastRequest = {
      days: Number(body.days) || 30,
      current_stock: Number(body.current_stock) ?? 5000,
      lead_time: Number(body.lead_time) || 5,
      ordering_cost: Number(body.ordering_cost) || 50,
      holding_cost: Number(body.holding_cost) || 2,
      safety_stock: Number(body.safety_stock) ?? 500,
      promo_boost_pct: Number(body.promo_boost_pct) || 0,
    };

    const result = runForecastSimulation(requestParams);
    return NextResponse.json(result);
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error?.message || 'Failed to process demand forecast' },
      { status: 500 }
    );
  }
}
