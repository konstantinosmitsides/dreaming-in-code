import json
import math

import networkx as nx
from pyvis.network import Network

# --- Comprehensive Color Map (Unchanged) ---
STATUS_COLOR_MAP = {
    # --- Generation States ---
    "seed": "#9B59B6",  # Purple
    "compile_success": "#48C9B0",  # Teal
    # --- Failure States ---
    "compile_fail": "#C0392B",  # Dark Red
    "compile_fail_no_code": "#C0392B",  # Dark Red
    "rejected_eval_fail": "#E74C3C",  # Lighter Red
    "rejected_no_episodes": "#E74C3C",  # Lighter Red
    # --- Active Set States ---
    "not_good_enough": "#95A5A6",  # Grey (Rejected from active set)
    "activated": "#3498DB",  # Blue (In active set, but status unknown)
    # --- Performance States (from categorization) ---
    "A": "#2ECC71",  # Green (Mastered, SR >= 0.75)
    "B": "#5DADE2",  # Light Blue (SR 0.5 - 0.75)
    "C": "#F1C40F",  # Yellow (SR 0.25 - 0.5)
    "D": "#E67E22",  # Carrot (SR < 0.25)
}
DEFAULT_COLOR = "#BDC3C7"  # Grey for any other status

# --- Layout Spacing (Unchanged) ---
Y_SPACING = 250
X_SPACING = 600

SEARCH_CONTROLS_HTML_CSS = """
<div id="search-controls">
    <input type="text" id="search-input" placeholder="Enter number...">
    <button id="search-btn">Search</button>
</div>
<style>
    #search-controls {
        position: absolute;
        top: 15px;
        left: 50%;
        transform: translateX(-50%);
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #d3d3d3;
        border-radius: 8px;
        padding: 8px 12px;
        z-index: 1002;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        display: flex;
        gap: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    #search-input {
        padding: 6px 10px;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 14px;
        width: 150px;
        outline: none;
    }
    #search-input:focus { border-color: #3498DB; }
    #search-btn {
        padding: 6px 12px;
        background-color: #3498DB;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 13px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    #search-btn:hover { background-color: #2980B9; }
</style>
"""

FILTER_CONTROLS_HTML_CSS = """
<div id="filter-controls">
    <h4>Graph Filters</h4>
    <div id="filter-buttons">
        <button id="filter-all" class="filter-button active">Show All</button>
        <button id="filter-active" class="filter-button">Active Set Only</button>
        <hr style="border:none;border-top:1px solid #e0e0e0;margin:10px 0;">
        <button id="filter-A" class="filter-button">Status A Only</button>
        <button id="filter-B" class="filter-button">Status B Only</button>
        <button id="filter-C" class="filter-button">Status C Only</button>
        <button id="filter-D" class="filter-button">Status D Only</button>
    </div>
</div>
<style>
    #filter-controls {
        position: absolute; 
        top: 15px; 
        right: 15px;
        width: 170px;
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #d3d3d3; 
        border-radius: 8px;
        padding: 10px 15px; 
        z-index: 1001;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
    #filter-controls h4 {
        margin-top: 0; 
        margin-bottom: 10px; 
        text-align: center; 
        font-weight: 600;
        font-size: 15px;
    }
    .filter-button {
        display: block;
        width: 100%;
        padding: 8px 10px;
        margin-bottom: 5px;
        font-size: 13px;
        font-weight: 500;
        border: 1px solid #ccc;
        background-color: #f9f9f9;
        color: #333;
        border-radius: 5px;
        cursor: pointer;
        text-align: left;
        transition: background-color 0.2s, color 0.2s;
    }
    .filter-button:last-child { margin-bottom: 0; }
    .filter-button:hover { background-color: #e9e9e9; }
    .filter-button.active {
        background-color: #3498DB;
        color: white;
        border-color: #3498DB;
    }
</style>
"""

# --- HTML & CSS for the Bottom "Inspector" Panel (Unchanged) ---
DETAILS_PANEL_HTML_CSS = """
<div id="resizer"></div>
<div id="details-panel">
    <h4 id="details-header">Node Inspector</h4>
    <div id="details-content">
        <p>Click on a node to inspect its properties.</p>
    </div>
</div>
<style>
    /* --- Main Layout Styling --- */
    html, body {
        margin: 0;
        padding: 0;
        height: 100%;
        overflow: hidden;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        background-color: #f0f2f5;
    }
    #mynetwork {
        height: 70%; /* Initial height of the graph */
        width: 100%;
        border-bottom: 2px solid #ccc;
        box-sizing: border-box; 
        background-color: #ffffff;
    }
    #resizer {
        height: 8px;
        background: #e0e0e0;
        cursor: ns-resize;
        border-top: 1px solid #ccc;
        border-bottom: 1px solid #ccc;
    }
    #details-panel {
        height: calc(30% - 10px); /* Initial height of the panel */
        width: 100%;
        background-color: #f9f9f9;
        display: flex;
        flex-direction: column;
    }
    #details-header {
        margin: 0;
        padding: 12px 15px;
        background-color: #efefef;
        border-bottom: 1px solid #d3d3d3;
        font-size: 16px;
        font-weight: 600;
        flex-shrink: 0; 
    }
    #details-content {
        padding: 15px;
        overflow-y: auto; /* Allow vertical scrolling */
        flex-grow: 1; 
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 14px;
    }
    #details-content p {
        margin-top: 0;
        color: #888;
        font-family: sans-serif;
    }
    
    /* --- Inspector Table Styling --- */
    .inspector-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 15px;
    }
    .inspector-table th, .inspector-table td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
        font-family: sans-serif;
    }
    .inspector-table th {
        width: 150px;
        font-weight: 600;
        color: #555;
        background-color: #f5f5f5;
    }
    .inspector-table td {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    }
    
    /* --- Section Header Styling --- */
    .inspector-section {
        font-family: sans-serif;
        font-size: 15px;
        font-weight: 600;
        color: #333;
        margin-top: 10px;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 2px solid #e0e0e0;
    }

    /* --- Link List Styling --- */
    .link-list {
        list-style: none;
        padding-left: 0;
        margin-top: 10px;
    }
    .link-list li {
        margin-bottom: 5px;
    }
    .link-list a {
        text-decoration: none;
        color: #2B7CE9;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    }
    .link-list a:hover {
        text-decoration: underline;
    }

    /* --- Code Block Styling --- */
    .code-block {
        background-color: #2d2d2d;
        color: #dcdcdc;
        padding: 15px;
        border-radius: 5px;
        overflow-x: auto;
        white-space: pre;
    }
    /* Simple keyword highlighting */
    .code-block .py-keyword { color: #569cd6; }
    .code-block .py-def { color: #569cd6; font-weight: bold; }
    .code-block .py-class { color: #4ec9b0; font-weight: bold; }
    .code-block .py-string { color: #ce9178; }
    .code-block .py-comment { color: #6a9955; }
    .code-block .py-number { color: #b5cea8; }

</style>
"""

JAVASCRIPT_CODE = """
<script type="text/javascript">
document.addEventListener("DOMContentLoaded", function() {
try {
    if (typeof network === 'undefined') {
        console.error("network is undefined (pyvis/vis.js not loaded?)");
        return;
    }

    // --- Get DOM Elements ---
    const graphContainer = document.getElementById('mynetwork');
    const detailsPanel = document.getElementById('details-panel');
    const detailsContent = document.getElementById('details-content');
    const panelHeader = document.getElementById('details-header');
    const resizer = document.getElementById('resizer');
    
    // --- Search Elements ---
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');

    const allNodes = network.body.data.nodes;
    const allEdges = network.body.data.edges.get();

    // --- Store default edge properties ---
    const defaultEdgeOptions = allEdges.map(edge => ({
        id: edge.id,
        color: { color: '#888888', opacity: 0.6 },
        width: 1
    }));

    // --- Cache Node Database for Filtering & Ghosting ---
    let nodeDatabase = [];
    
    try {
        const allNodeIds = allNodes.getIds();
        allNodeIds.forEach(id => {
            const node = allNodes.get(id);
            if (node && node.nodeData) {
                let nodeData = JSON.parse(node.nodeData);
                nodeData.originalLabel = node.label;
                nodeData.originalColor = node.color;
                nodeData.originalBorderWidth = node.borderWidth;
                nodeData.originalFont = node.font || { color: '#343434' }; 
                nodeDatabase.push(nodeData);
            }
        });
    } catch (e) {
        console.error("Failed to build node database:", e);
    }

    // --- Filter Logic (Ghosting) ---
    const ghostColor = { border: '#e0e0e0', background: '#f9f9f9' };
    const ghostFont = { color: '#cccccc' };

    function applyFilter(filterType) {
        let updates = [];
        nodeDatabase.forEach(node => {
            let isGhosted = false;
            switch (filterType) {
                case 'active': isGhosted = !node.is_active; break;
                case 'A':
                case 'B':
                case 'C':
                case 'D': isGhosted = node.status !== filterType; break;
                case 'all':
                default: isGhosted = false; break;
            }

            if (isGhosted) {
                updates.push({
                    id: node.id, label: node.id, color: ghostColor,
                    font: ghostFont, borderWidth: 1
                });
            } else {
                updates.push({
                    id: node.id, label: node.originalLabel, color: node.originalColor,
                    font: node.originalFont, borderWidth: node.originalBorderWidth
                });
            }
        });
        allNodes.update(updates);
    }

    // --- Filter Button Listeners ---
    const filterButtons = document.querySelectorAll('.filter-button');
    filterButtons.forEach(button => {
        button.addEventListener('click', function() {
            filterButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            const filterType = this.id.split('-')[1];
            applyFilter(filterType);
        });
    });

    // --- NEW: Search Logic (Smart ID Mapping) ---
    function executeSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        let targetId = null;

        // 1. Try exact match (e.g., user typed "task_5" or just "5" if ID is "5")
        if (allNodes.get(query)) {
            targetId = query;
        } 
        // 2. Try adding "task_" prefix (e.g., user typed "5", we check "task_5")
        else if (allNodes.get("task_" + query)) {
            targetId = "task_" + query;
        }

        if (targetId) {
            // 1. Select the node visually
            network.selectNodes([targetId]);
            
            // 2. Focus/Zoom animation
            network.focus(targetId, {
                scale: 1.2,
                animation: {
                    duration: 1000,
                    easingFunction: "easeInOutQuad"
                }
            });

            // 3. Manually trigger the selectNode logic to update Inspector
            network.emit('selectNode', { nodes: [targetId] });
        } else {
            alert('Task ID "' + query + '" (or "task_' + query + '") not found in the graph.');
        }
    }

    if(searchBtn && searchInput) {
        searchBtn.addEventListener('click', executeSearch);
        searchInput.addEventListener('keyup', function(e) {
            if (e.key === 'Enter') executeSearch();
        });
    }

    // --- Resizer Logic ---
    let isResizing = false;
    resizer.addEventListener('mousedown', function(e) {
        isResizing = true;
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', function() {
            isResizing = false;
            document.removeEventListener('mousemove', handleMouseMove);
            network.setSize('100%', '100%');
        }, { once: true });
    });

    function handleMouseMove(e) {
        if (!isResizing) return;
        const totalHeight = window.innerHeight;
        const newGraphHeight = e.clientY;
        const newPanelHeight = totalHeight - newGraphHeight - resizer.offsetHeight;
        if (newGraphHeight > 100 && newPanelHeight > 100) {
            graphContainer.style.height = `${newGraphHeight}px`;
            detailsPanel.style.height = `${newPanelHeight}px`;
        }
    }

    // --- Network Interaction Logic ---
    network.on("selectNode", function(params) {
        const selectedNodeId = params.nodes[0];
        if (!selectedNodeId) return;

        const node = allNodes.get(selectedNodeId);
        let nodeData = {};
        if (node && node.nodeData) {
            try { nodeData = JSON.parse(node.nodeData); } 
            catch (e) { detailsContent.innerHTML = "<p>Error parsing data</p>"; return; }
        } else {
            detailsContent.innerHTML = "<p>No node data available.</p>"; return;
        }

        panelHeader.textContent = `Inspector: ${nodeData.id}`;
        detailsContent.innerHTML = buildInspectorHTML(nodeData);
        detailsContent.scrollTop = 0;

        const connectedEdgeIds = network.getConnectedEdges(selectedNodeId);
        const edgeUpdates = allEdges.map(edge => {
            if (connectedEdgeIds.includes(edge.id)) return { id: edge.id, color: '#2B7CE9', width: 2.5 };
            else return { id: edge.id, color: '#E0E0E0', width: 0.5 };
        });
        network.body.data.edges.update(edgeUpdates);
    });

    network.on("deselectNode", function(params) {
        panelHeader.textContent = 'Node Inspector';
        detailsContent.innerHTML = '<p>Click on a node to inspect its properties.</p>';
        network.body.data.edges.update(defaultEdgeOptions);
    });

    detailsContent.addEventListener('click', function(e) {
        if (e.target && e.target.classList.contains('node-link')) {
            e.preventDefault();
            const newNodeId = e.target.getAttribute('data-node-id');
            if (newNodeId) {
                network.selectNodes([newNodeId]);
                network.emit('selectNode', { nodes: [newNodeId] });
                network.focus(newNodeId, { scale: 1.0 });
            }
        }
    });

    // --- HTML Builders ---
    function buildInspectorHTML(data) {
        let sr = (data.latest_sr != null && data.latest_sr >= 0) ? `${(data.latest_sr * 100).toFixed(1)}%` : 'N/A';
        let status = data.status || 'N/A';
        let p_score = (data.priority_score != null) ? Number(data.priority_score).toFixed(4) : 'N/A';
        let lastTrained = 'Never';
        if (data.session_last_trained) lastTrained = `Session ${data.session_last_trained}`;

        let statsHTML = `
            <div class="inspector-section">📊 Key Stats</div>
            <table class="inspector-table">
            <tr><th>Status</th> <td>${escapeHTML(status)}</td></tr>
            <tr><th>Latest SR</th> <td>${sr}</td></tr>
            <tr><th>Priority Score</th> <td>${p_score}</td></tr>
            <tr><th>Is Active?</th> <td>${data.is_active ? '✅ Yes' : '❌ No'}</td></tr>
            </table>`;

        let detailsHTML = `
            <div class="inspector-section">📋 Details</div>
            <table class="inspector-table">
            <tr><th>Node ID</th> <td>${escapeHTML(data.id)}</td></tr>
            <tr><th>Type</th> <td>${escapeHTML(data.type)}</td></tr>
            <tr><th>Created</th> <td>Session ${escapeHTML(data.session_created)}</td></tr>
            <tr><th>Last Trained</th> <td>${escapeHTML(lastTrained)}</td></tr>
            </table>`;

        let parentsHTML = '<div class="inspector-section">🧬 Parents</div>';
        if (data.parents && data.parents.length > 0) {
            parentsHTML += `<ul class="link-list">${data.parents.map(id => `<li><a href="#" class="node-link" data-node-id="${escapeHTML(id)}">${escapeHTML(id)}</a></li>`).join('')}</ul>`;
        } else { parentsHTML += '<p style="color:#888;">Root node.</p>'; }

        let childrenHTML = '<div class="inspector-section"> Offspring</div>';
        if (data.children && data.children.length > 0) {
            childrenHTML += `<ul class="link-list">${data.children.map(id => `<li><a href="#" class="node-link" data-node-id="${escapeHTML(id)}">${escapeHTML(id)}</a></li>`).join('')}</ul>`;
        } else { childrenHTML += '<p style="color:#888;">No children.</p>'; }

        let historyHTML = '';
        if (data.performance_history && data.performance_history.length > 0) {
             historyHTML = `<div class="inspector-section">📈 Performance History</div><ul>${data.performance_history.map(item => `<li>Session ${item.session}: SR ${item.sr != null ? (item.sr*100).toFixed(1)+'%' : 'N/A'}</li>`).join('')}</ul>`;
        }

        let descHTML = `<div class="inspector-section">📝 Description</div><pre style="white-space: pre-wrap; font-family: sans-serif;">${escapeHTML(data.description)}</pre>`;
        let reasHTML = `<div class="inspector-section">🧠 Reasoning</div><pre style="white-space: pre-wrap; font-family: sans-serif;">${escapeHTML(data.reasoning)}</pre>`;
        let codeHTML = data.code_string ? `<div class="inspector-section">🐍 Code</div><div class="code-block">${syntaxHighlight(data.code_string)}</div>` : '';

        return statsHTML + detailsHTML + parentsHTML + childrenHTML + reasHTML + historyHTML + descHTML + codeHTML;
    }

    function escapeHTML(str) { return String(str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
    
    function syntaxHighlight(code) {
        code = escapeHTML(code);
        return code.replace(/#.*$/gm, m => `<span class="py-comment">${m}</span>`)
                   .replace(/(".*?"|'.*?')/g, m => `<span class="py-string">${m}</span>`)
                   .replace(/\b(def|class|if|else|return|import|from)\b/g, m => `<span class="py-keyword">${m}</span>`);
    }

} catch (e) { console.error("Graph UI script crashed:", e); }
});
</script>
"""


# # --- Helper function for Legend (Unchanged) ---
# def create_legend_html(color_map: dict) -> str:
#     legend_items = ""
#     unique_colors = {}
#     for status, color in color_map.items():
#         if color not in unique_colors:
#             unique_colors[color] = []
#         unique_colors[color].append(status.capitalize())

#     for color, statuses in unique_colors.items():
#         label = " / ".join(statuses)
#         legend_items += f'<li><span class="legend-color-box" style="background-color:{color};"></span><span class="legend-label">{label}</span></li>'

#     # --- Add the 'Active Set' border to the legend ---
#     legend_items += '<li><span class="legend-color-box" style="border: 4px solid #8E44AD; background-color: #fff;"></span><span class="legend-label">In Active Set</span></li>'

#     return f"""
#     <div id="graph-legend"><h4>Node Status</h4><ul>{legend_items}</ul></div>
#     <style>
#         #graph-legend {{ 
#             position: absolute; top: 15px; left: 15px; 
#             background-color: rgba(255, 255, 255, 0.95); 
#             border: 1px solid #d3d3d3; border-radius: 8px; 
#             padding: 10px 15px; z-index: 1000; 
#             box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
#             font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
#             font-size: 14px;
#         }}
#         #graph-legend h4 {{ margin-top: 0; margin-bottom: 10px; text-align: center; font-weight: 600; }}
#         #graph-legend ul {{ list-style: none; padding: 0; margin: 0; }}
#         #graph-legend li {{ display: flex; align-items: center; margin-bottom: 5px; }}
#         .legend-color-box {{ 
#             width: 20px; 
#             height: 20px; 
#             margin-right: 10px; 
#             border: 1px solid #ccc; 
#             border-radius: 3px; 
#             box-sizing: border-box; /* Ensures border is inside the box */
#         }}
#     </style>
#     """
def create_legend_html(color_map: dict, label_map: dict | None = None) -> str:
    legend_items = ""

    for status, color in color_map.items():
        if label_map and status in label_map:
            label = label_map[status]
        else:
            label = status.capitalize()

        legend_items += (
            f'<li>'
            f'<span class="legend-color-box" style="background-color:{color};"></span>'
            f'<span class="legend-label">{label}</span>'
            f'</li>'
        )

    # --- Active Set border legend ---
    legend_items += (
        '<li>'
        '<span class="legend-color-box" '
        'style="border: 4px solid #8E44AD; background-color: #fff;"></span>'
        '<span class="legend-label">In Active Set</span>'
        '</li>'
    )

    return f"""
    <div id="graph-legend">
        <h4>Node Status</h4>
        <ul>{legend_items}</ul>
    </div>
    <style>
        #graph-legend {{ 
            position: absolute; top: 15px; left: 15px; 
            background-color: rgba(255, 255, 255, 0.95); 
            border: 1px solid #d3d3d3; border-radius: 8px; 
            padding: 10px 15px; z-index: 1000; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-size: 14px;
        }}
        #graph-legend h4 {{
            margin-top: 0;
            margin-bottom: 10px;
            text-align: center;
            font-weight: 600;
        }}
        #graph-legend ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        #graph-legend li {{
            display: flex;
            align-items: center;
            margin-bottom: 6px;
        }}
        .legend-color-box {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border: 1px solid #ccc;
            border-radius: 3px;
            box-sizing: border-box;
        }}
    </style>
    """


# --- Top-level helper function to sanitize floats (Unchanged) ---
def _sanitize_float_for_json(value):
    """Converts nan and inf to None for safe JSON serialization."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


# --- Main Function (UPDATED) ---
def create_graph_visualization_html(graphml_path: str) -> str:
    """Reads a GraphML file and generates a self-contained, interactive
    pyvis HTML visualization as a string.
    - Rich "Inspector" panel
    - Border for active nodes (now purple)
    - Wider layout
    - Disables physics and default highlighting
    - Clickable parent/child links in inspector
    - Fix: Handles NaN/Inf values for safe JSON parsing
    - Fix: New color for compile_success
    - NEW: Filter buttons
    """
    print(f"Reading graph from {graphml_path}...")
    try:
        G = nx.read_graphml(graphml_path)

        # --- NEW: Keep only trained nodes (A/B/C/D) ---
        TRAINED_STATUSES = {"A", "B", "C", "D"}
        keep_nodes = [n for n, d in G.nodes(data=True) if d.get("status", "unknown") in TRAINED_STATUSES]
        G = G.subgraph(keep_nodes).copy()
    except Exception as e:
        print(f"Error reading GraphML file: {e}")
        return f"<html><body><h1>Error reading graph file</h1><p>{e}</p></body></html>"

    print("Processing graph nodes for visualization...")
    for node, data in G.nodes(data=True):
        # --- 1. Extract All Data (Unchanged) ---
        status = data.get("status", "unknown")
        node_type = data.get("type", "generated")
        description = data.get("description", "No description.")
        code_string = data.get("code")
        session_created = int(data.get("session_created", 0))
        session_last_trained = int(data.get("session_last_trained", 0))
        priority_score = float(data.get("priority_score", 0.0))
        is_active = bool(data.get("is_active", False))
        reasoning = data.get("reasoning", "No reasoning provided.")

        # --- 2. Safely Parse Performance History (Unchanged) ---
        latest_sr = -1.0
        performance_history_list = []
        history_str = data.get("performance_history")

        if history_str and isinstance(history_str, str):
            try:
                safe_history_str = (
                    history_str.replace("NaN", "null")
                    .replace("Infinity", "null")
                    .replace("-Infinity", "null")
                )
                performance_history_list = json.loads(safe_history_str)
                if performance_history_list:
                    latest_performance = performance_history_list[-1]
                    latest_sr = float(latest_performance.get("sr", -1.0))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode performance_history for node {node}")
                performance_history_list = []

        # --- 3. Set Visual Properties (Unchanged) ---
        label = str(node)
        safe_latest_sr_label = _sanitize_float_for_json(latest_sr)
        if safe_latest_sr_label is not None and safe_latest_sr_label >= 0:
            sr_percent = f"{safe_latest_sr_label:.0%}"
            label = f"{node}\nSR: {sr_percent}"

        G.nodes[node]["label"] = label
        G.nodes[node]["shape"] = "box" if node_type == "seed" else "ellipse"
        G.nodes[node]["level"] = session_created

        node_fill_color = STATUS_COLOR_MAP.get(status, DEFAULT_COLOR)

        if is_active:
            G.nodes[node]["color"] = {"border": "#8E44AD", "background": node_fill_color}
            G.nodes[node]["borderWidth"] = 1.5
        else:
            G.nodes[node]["color"] = node_fill_color
            G.nodes[node]["borderWidth"] = 1

        tooltip_sr = f"{safe_latest_sr_label:.1%}" if safe_latest_sr_label is not None else "N/A"
        G.nodes[node]["title"] = (
            f"Task ID: {node}\nStatus: {status}\nSR: {tooltip_sr}\nActive: {is_active}"
        )

        # --- 4. Bundle ALL data for the Inspector Panel (Unchanged) ---
        safe_latest_sr = _sanitize_float_for_json(latest_sr)
        safe_priority_score = _sanitize_float_for_json(priority_score)

        safe_performance_history = []
        for item in performance_history_list:
            safe_item = item.copy()
            if "sr" in safe_item:
                safe_item["sr"] = _sanitize_float_for_json(safe_item.get("sr"))
            if "lp" in safe_item:
                safe_item["lp"] = _sanitize_float_for_json(safe_item.get("lp"))
            safe_performance_history.append(safe_item)

        node_data_payload = {
            "id": str(node),
            "status": status,
            "type": node_type,
            "latest_sr": safe_latest_sr,
            "priority_score": safe_priority_score,
            "is_active": is_active,
            "session_created": session_created,
            "session_last_trained": session_last_trained,
            "performance_history": safe_performance_history,
            "description": description,
            "code_string": code_string,
            "reasoning": reasoning,
            "parents": [str(p) for p in G.predecessors(node)],
            "children": [str(s) for s in G.successors(node)],
        }

        try:
            G.nodes[node]["nodeData"] = json.dumps(node_data_payload)
        except Exception as e:
            print(f"Error serializing node data for {node}: {e}")
            G.nodes[node]["nodeData"] = json.dumps(
                {"id": str(node), "error": "Serialization failed"}
            )

    # 3. Create a Pyvis network
    print("Initializing Pyvis network...")
    net = Network(
        height="100%", width="100%", notebook=False, directed=True, cdn_resources="in_line"
    )
    net.from_nx(G)

    # 4. Set hierarchical layout options (Unchanged)
    print("Setting network layout and physics options...")
    net.set_options("""
    var options = {
    "layout": {
        "hierarchical": {
        "enabled": true,
        "direction": "UD",
        "sortMethod": "directed",
        "levelSeparation": 200,
        "nodeSpacing": 600, 
        "treeSpacing": 800
        }
    },
    "interaction": { 
        "hover": true, 
        "tooltipDelay": 200,
        "navigationButtons": true,
        "keyboard": true,
        "highlightNearest": { "enabled": false } 
    },
    "nodes": { 
        "font": { "size": 16, "face": "sans-serif" }
    },
    "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 }},
        "color": { "color": "#888888", "opacity": 0.6 },
        "smooth": { "type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4 }
    },
    "physics": {
        "enabled": false
    }
    }
    """)

    # 5. Generate and embellish the HTML content (UPDATED)
    print("Generating final HTML string...")
    html_content = net.generate_html(notebook=False)

    # LEGEND_STATUSES = ["A", "B", "C", "D"]
    # legend_color_map = {k: STATUS_COLOR_MAP[k] for k in LEGEND_STATUSES if k in STATUS_COLOR_MAP}
    # legend_html = create_legend_html(legend_color_map)

    STATUS_LEGEND_LABELS = {
        "A": "A (SR ≥ 0.75)",
        "B": "B (0.50 ≤ SR < 0.75)",
        "C": "C (0.25 ≤ SR < 0.50)",
        "D": "D (SR < 0.25)",
    }

    LEGEND_STATUSES = ["A", "B", "C", "D"]
    legend_color_map = {k: STATUS_COLOR_MAP[k] for k in LEGEND_STATUSES}

    legend_html = create_legend_html(
        legend_color_map,
        STATUS_LEGEND_LABELS
    )

    network_div_end = html_content.rfind("</div>", 0, html_content.rfind("<script")) + len("</div>")
    html_with_panel = (
        html_content[:network_div_end] + DETAILS_PANEL_HTML_CSS + html_content[network_div_end:]
    )

    body_tag_start = html_with_panel.find("<body>") + len("<body>")
    # --- NEW: Inject filter controls AND legend ---
    html_with_addons = (
        html_with_panel[:body_tag_start]
        + FILTER_CONTROLS_HTML_CSS  
        + SEARCH_CONTROLS_HTML_CSS  
        + legend_html
        + html_with_panel[body_tag_start:]
    )

    final_html = html_with_addons.replace("</body>", f"{JAVASCRIPT_CODE}</body>")

    print("✅ HTML generation complete.")
    return final_html


# (Safe test block for your `logging_utils.py`)
if __name__ == "__main__":
    GRAPHML_FILE = "/home_nfs/konstantinos/projects/DiCode/personal_data/paper_results/DICODE/31249/task_graph.graphml"
    OUTPUT_FILE = "personal_data/html_archives/run_5_archive.html"

    final_html = create_graph_visualization_html(GRAPHML_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_html)

    print("✅ Successfully created visualization with 'Active Set' borders!")
    print(f"   Please open '{OUTPUT_FILE}' in your web browser.")
