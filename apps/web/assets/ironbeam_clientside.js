if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.ironbeam = {
    sync_crosshair_and_zoom: function(relayoutData, hoverData, fig) {
        if (!fig) return window.dash_clientside.no_update;

        // Optimization: Shallow copy instead of deep clone to improve performance
        // We only modify layout.xaxis and layout.shapes
        let new_fig = Object.assign({}, fig);
        new_fig.layout = Object.assign({}, fig.layout);
        
        // Ensure xaxis exists and is a new object
        if (new_fig.layout.xaxis) {
            new_fig.layout.xaxis = Object.assign({}, new_fig.layout.xaxis);
        } else {
            new_fig.layout.xaxis = {};
        }

        let updated = false;

        // --- 1. Sync Zoom/Pan (X-axis) ---
        if (relayoutData) {
            let x0, x1;
            // Handle different shapes of relayoutData
            if (relayoutData['xaxis.range[0]'] !== undefined && relayoutData['xaxis.range[1]'] !== undefined) {
                x0 = relayoutData['xaxis.range[0]'];
                x1 = relayoutData['xaxis.range[1]'];
            } else if (relayoutData['xaxis.range'] && Array.isArray(relayoutData['xaxis.range']) && relayoutData['xaxis.range'].length === 2) {
                x0 = relayoutData['xaxis.range'][0];
                x1 = relayoutData['xaxis.range'][1];
            }
            
            if (x0 !== undefined && x1 !== undefined) {
                new_fig.layout.xaxis.range = [x0, x1];
                new_fig.layout.xaxis.autorange = false;
                updated = true;
            }
            
            if (relayoutData['xaxis.autorange'] === true) {
                new_fig.layout.xaxis.autorange = true;
                updated = true;
            }
        }

        // --- 2. Sync Crosshair (Vertical Line) ---
        let x_val = null;
        let show_crosshair = false;
        
        if (hoverData && hoverData.points && hoverData.points[0] && hoverData.points[0].x) {
            x_val = hoverData.points[0].x;
            show_crosshair = true;
        }
        
        // Copy shapes array if we are going to modify it
        let shapes = new_fig.layout.shapes ? [...new_fig.layout.shapes] : [];
        let crosshair_idx = -1;
        
        // Find existing crosshair (dashed line)
        for (let i = 0; i < shapes.length; i++) {
            const s = shapes[i];
            if (s.type === 'line' && s.yref === 'paper' && s.line && s.line.dash === 'dash') {
                crosshair_idx = i;
                break;
            }
        }
        
        if (show_crosshair) {
            const new_shape = {
                type: 'line',
                x0: x_val,
                x1: x_val,
                y0: 0,
                y1: 1,
                xref: 'x',
                yref: 'paper',
                line: {
                    color: 'rgba(255,255,255,0.35)',
                    width: 1,
                    dash: 'dash'
                }
            };
            
            if (crosshair_idx >= 0) {
                shapes[crosshair_idx] = new_shape;
            } else {
                shapes.push(new_shape);
            }
            updated = true;
        } else {
            // Hide crosshair if it exists
            if (crosshair_idx >= 0) {
                // Make it transparent
                shapes[crosshair_idx] = Object.assign({}, shapes[crosshair_idx]);
                shapes[crosshair_idx].line = Object.assign({}, shapes[crosshair_idx].line);
                shapes[crosshair_idx].line.color = 'rgba(0,0,0,0)';
                updated = true;
            }
        }
        
        new_fig.layout.shapes = shapes;

        return updated ? new_fig : window.dash_clientside.no_update;
    },

    highlight_candle: function(clickData, fig) {
        if (!clickData || !clickData.points || !clickData.points.length || !fig) {
            return window.dash_clientside.no_update;
        }

        const pt = clickData.points[0];
        const curveIdx = pt.curveNumber;
        const pointIdx = pt.pointNumber;
        
        // Check if valid trace
        if (!fig.data || !fig.data[curveIdx]) return window.dash_clientside.no_update;

        const sourceTrace = fig.data[curveIdx];
        // Ensure it's a candlestick or has the data we need
        if (!sourceTrace.open || !sourceTrace.close || !sourceTrace.x) {
            return window.dash_clientside.no_update;
        }

        const x_val = sourceTrace.x[pointIdx];
        const o_val = sourceTrace.open[pointIdx];
        const h_val = sourceTrace.high[pointIdx];
        const l_val = sourceTrace.low[pointIdx];
        const c_val = sourceTrace.close[pointIdx];

        // Shallow copy fig and data array
        let new_fig = Object.assign({}, fig);
        new_fig.data = [...fig.data];

        // Find "Selected slices" trace
        let selTraceIdx = -1;
        for (let i = 0; i < new_fig.data.length; i++) {
            if (new_fig.data[i].name === "Selected slices") {
                selTraceIdx = i;
                break;
            }
        }

        if (selTraceIdx === -1) {
            // Create new trace
            // Use the same color as defined in python: HIGHLIGHT_COLOR = "#ef4444"
            const newTrace = {
                type: 'candlestick',
                x: [x_val],
                open: [o_val],
                high: [h_val],
                low: [l_val],
                close: [c_val],
                name: 'Selected slices',
                increasing: {line: {color: '#ef4444', width: 2.0}, fillcolor: '#ef4444'},
                decreasing: {line: {color: '#ef4444', width: 2.0}, fillcolor: '#ef4444'},
                showlegend: false,
                yaxis: 'y2',
                hovertemplate: '<extra></extra>'
            };
            new_fig.data.push(newTrace);
        } else {
            // Update existing trace
            let selTrace = Object.assign({}, new_fig.data[selTraceIdx]);
            // Clone arrays
            selTrace.x = selTrace.x ? [...selTrace.x] : [];
            selTrace.open = selTrace.open ? [...selTrace.open] : [];
            selTrace.high = selTrace.high ? [...selTrace.high] : [];
            selTrace.low = selTrace.low ? [...selTrace.low] : [];
            selTrace.close = selTrace.close ? [...selTrace.close] : [];

            const existIdx = selTrace.x.indexOf(x_val);
            
            if (existIdx >= 0) {
                // Remove
                selTrace.x.splice(existIdx, 1);
                selTrace.open.splice(existIdx, 1);
                selTrace.high.splice(existIdx, 1);
                selTrace.low.splice(existIdx, 1);
                selTrace.close.splice(existIdx, 1);
            } else {
                // Add
                selTrace.x.push(x_val);
                selTrace.open.push(o_val);
                selTrace.high.push(h_val);
                selTrace.low.push(l_val);
                selTrace.close.push(c_val);
            }
            
            new_fig.data[selTraceIdx] = selTrace;
        }

        return new_fig;
    }
};
