import { NextRequest, NextResponse } from 'next/server';
import * as XLSX from 'xlsx';
import { PORequest } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  try {
    const body: PORequest = await req.json();

    const poNumber = `PO-${Date.now().toString().slice(-8)}`;
    const poDate = new Date().toISOString().split('T')[0];
    const totalCost = (body.recommended_qty || 0) * (body.unit_cost || 0);

    const wb = XLSX.utils.book_new();

    // Sheet data array of arrays
    const sheetData = [
      ['ENTERPRISE PURCHASE ORDER', '', '', '', ''],
      ['', '', '', '', ''],
      ['PO Number:', poNumber, '', 'Date:', poDate],
      ['Vendor / Supplier:', body.vendor_name || 'Global Logistics Supply Ltd', '', 'Lead Time:', `${body.lead_time_days} Days`],
      ['Order Status:', 'APPROVED / PENDING TRANSMISSION', '', 'Created By:', 'AI Inventory Engine'],
      ['', '', '', '', ''],
      ['LINE ITEMS & INVENTORY METRICS', '', '', '', ''],
      [
        'Item Description',
        'Current Stock (Units)',
        'Reorder Point (ROP)',
        'Optimal EOQ',
        'Avg Daily Demand',
        'Order Qty (Units)',
        'Unit Price (INR)',
        'Subtotal (INR)'
      ],
      [
        body.item_name || 'Aggregated SKU #1',
        Number(body.current_stock) || 0,
        Number(body.rop) || 0,
        Number(body.eoq) || 0,
        Number(body.avg_daily_demand) || 0,
        Number(body.recommended_qty) || 0,
        Number(body.unit_cost) || 0,
        totalCost
      ],
      ['', '', '', '', '', '', '', ''],
      ['SUMMARY TOTALS', '', '', '', '', '', '', ''],
      ['Total Units Ordered:', Number(body.recommended_qty) || 0, '', '', '', '', 'Total Order Value (INR):', totalCost],
    ];

    const ws = XLSX.utils.aoa_to_sheet(sheetData);

    // Set column widths
    ws['!cols'] = [
      { wch: 28 },
      { wch: 22 },
      { wch: 20 },
      { wch: 18 },
      { wch: 18 },
      { wch: 18 },
      { wch: 18 },
      { wch: 20 }
    ];

    XLSX.utils.book_append_sheet(wb, ws, 'Purchase Order');

    const excelBuffer = XLSX.write(wb, { type: 'buffer', bookType: 'xlsx' });
    const filename = `Purchase_Order_${(body.vendor_name || 'Vendor').replace(/\s+/g, '_')}_${poNumber}.xlsx`;

    return new NextResponse(excelBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'Content-Disposition': `attachment; filename="${filename}"`,
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error?.message || 'Failed to generate Purchase Order' },
      { status: 500 }
    );
  }
}
