if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.ironbeam = {
    sync_crosshair_and_zoom: function(relayoutData, hoverData, fig) {
        if (!fig) return window.dash_clientside.no_update;

        // Deep copy layout to avoid mutating the state directly
        let new_fig = JSON.parse(JSON.stringify(fig));
        if (!new_fig.layout) new_fig.layout = {};
        if (!new_fig.layout.xaxis) new_fig.layout.xaxis = {};
        
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
        
        let shapes = new_fig.layout.shapes || [];
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
                shapes[crosshair_idx].line.color = 'rgba(0,0,0,0)';
                updated = true;
            }
        }
        
        new_fig.layout.shapes = shapes;

        return updated ? new_fig : window.dash_clientside.no_update;
    }
};
