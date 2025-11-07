/**
 * ComfyUI-Grounding Dynamic Parameter Management
 *
 * This extension manages the visibility of model-specific parameters
 * based on the selected model in both GroundingModelLoader and GroundingDetector nodes.
 */

import { app } from "../../../scripts/app.js";

const DEBUG = true; // Set to true to enable console logging

function log(...args) {
    if (DEBUG) {
        console.log("[ComfyUI-Grounding]", ...args);
    }
}

// Helper function to detect model type from model name
function getModelType(modelName) {
    if (modelName.startsWith("GroundingDINO:") || modelName.startsWith("MM-GroundingDINO:")) {
        return "grounding_dino";
    }
    if (modelName.startsWith("OWLv2:")) {
        return "owlv2";
    }
    if (modelName.startsWith("Florence-2:")) {
        return "florence2";
    }
    if (modelName.startsWith("YOLO-World:")) {
        return "yolo_world";
    }
    return "unknown";
}

// Helper function to hide a widget (NUCLEAR OPTION: removes from array)
function hideWidget(node, widget, suffix = "") {
    if (!widget) return;

    // If already hidden, skip
    if (widget._hidden) {
        log("Widget already hidden:", widget.name);
        return;
    }

    log("Hiding widget:", widget.name, "current type:", widget.type);

    // Find widget's current index in the widgets array
    const index = node.widgets.indexOf(widget);
    if (index === -1) {
        log("  ERROR: Widget not found in node.widgets array!");
        return;
    }

    log("  Found widget at index:", index);

    // Store original properties for restoration
    if (!widget.origType) {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
        widget.origSerializeValue = widget.serializeValue;
    }

    // Store the original index and mark as hidden
    widget._originalIndex = index;
    widget._hidden = true;

    // NUCLEAR OPTION: Remove widget from the array entirely
    node.widgets.splice(index, 1);

    log("  Widget removed from array at index:", index);
    log("  Remaining widgets:", node.widgets.length);

    // Hide linked widgets if any
    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach(w => hideWidget(node, w, ":hidden"));
    }
}

// Helper function to show a widget (NUCLEAR OPTION: re-inserts into array)
function showWidget(node, widget) {
    if (!widget) return;

    // If not hidden, skip
    if (!widget._hidden) {
        log("Widget already visible:", widget.name);
        return;
    }

    log("Showing widget:", widget.name);

    // Restore original properties
    if (widget.origType) {
        widget.type = widget.origType;
        widget.computeSize = widget.origComputeSize;

        if (widget.origSerializeValue) {
            widget.serializeValue = widget.origSerializeValue;
        }
    }

    // NUCLEAR OPTION: Re-insert widget into array at original position
    const targetIndex = widget._originalIndex;

    // Make sure we don't insert beyond array bounds
    const insertIndex = Math.min(targetIndex, node.widgets.length);

    node.widgets.splice(insertIndex, 0, widget);

    log("  Widget restored to array at index:", insertIndex);
    log("  Total widgets:", node.widgets.length);

    // Clear hidden flag
    widget._hidden = false;

    // Show linked widgets if any
    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach(w => showWidget(node, w));
    }
}

// Main extension registration
app.registerExtension({
    name: "comfyui.grounding.dynamic_parameters",

    async nodeCreated(node) {
        // Handle GroundingModelLoader node
        if (node.comfyClass === "GroundingModelLoader") {
            // Use setTimeout to ensure widgets are fully initialized
            // Increased delay to 100ms for better reliability across different systems
            setTimeout(() => {
                this.setupModelLoader(node);
            }, 100);
        }

        // Handle GroundingDetector node
        if (node.comfyClass === "GroundingDetector") {
            setTimeout(() => {
                this.setupDetector(node);
            }, 100);
        }

        // Handle GroundingMaskModelLoader node
        if (node.comfyClass === "GroundingMaskModelLoader") {
            setTimeout(() => {
                this.setupMaskModelLoader(node);
            }, 100);
        }

        // Handle GroundingMaskDetector node
        if (node.comfyClass === "GroundingMaskDetector") {
            setTimeout(() => {
                this.setupMaskDetector(node);
            }, 100);
        }
    },

    setupModelLoader(node) {
        log("Setting up GroundingModelLoader node");

        // Find the model widget
        const modelWidget = node.widgets?.find(w => w.name === "model");
        if (!modelWidget) {
            log("ERROR: Model widget not found!");
            log("Available widgets:", node.widgets?.map(w => w.name));
            return;
        }

        log("Model widget found, current value:", modelWidget.value);

        // Find all parameter widgets (load-time only)
        const groundingDinoAttnWidget = node.widgets?.find(w => w.name === "grounding_dino_attn");
        const florence2AttnWidget = node.widgets?.find(w => w.name === "florence2_attn");

        log("Found parameter widgets:", {
            grounding_dino_attn: !!groundingDinoAttnWidget,
            florence2_attn: !!florence2AttnWidget
        });

        // Function to update widget visibility based on selected model
        const updateWidgetVisibility = (selectedModel) => {
            const modelType = getModelType(selectedModel);
            log("Updating widget visibility for model:", selectedModel, "type:", modelType);

            // GroundingDINO parameters (load-time)
            if (groundingDinoAttnWidget) {
                if (modelType === "grounding_dino") {
                    showWidget(node, groundingDinoAttnWidget);
                } else {
                    hideWidget(node, groundingDinoAttnWidget);
                }
            }

            // Florence-2 parameters (load-time)
            if (florence2AttnWidget) {
                if (modelType === "florence2") {
                    showWidget(node, florence2AttnWidget);
                } else {
                    hideWidget(node, florence2AttnWidget);
                }
            }

            // Force canvas redraw immediately to update hidden state
            node.setDirtyCanvas(true, true);
            if (app.graph) {
                app.graph.setDirtyCanvas(true, true);
            }

            // CRITICAL: Delay size recalculation to prevent last widget from re-appearing
            // This ensures widget hiding completes before layout recalculation
            requestAnimationFrame(() => {
                // Recalculate node size after widgets are fully hidden
                const newSize = node.computeSize();
                node.setSize([node.size[0], newSize[1]]);

                // Mark as dirty again after resize
                node.setDirtyCanvas(true, true);
                if (app.canvas) {
                    app.canvas.setDirty(true, true);
                }

                // Final redraw after everything settles
                requestAnimationFrame(() => {
                    if (app.canvas) {
                        app.canvas.draw(true, true);
                    }
                    log("Widget visibility update complete");
                });
            });
        };

        // Store original callback
        const origCallback = modelWidget.callback;

        // Override callback to update visibility when model changes
        modelWidget.callback = function(value) {
            log("Model changed to:", value);

            // Call original callback if it exists
            const result = origCallback?.apply(this, arguments);

            // Update widget visibility
            updateWidgetVisibility(value);

            return result;
        };

        // Initialize visibility on node creation
        log("Initializing widget visibility...");
        updateWidgetVisibility(modelWidget.value);
    },

    setupDetector(node) {
        log("Setting up GroundingDetector node");

        // Find all model-specific parameter widgets (inference-time only)
        const textThresholdWidget = node.widgets?.find(w => w.name === "text_threshold");
        const florence2MaxTokensWidget = node.widgets?.find(w => w.name === "florence2_max_tokens");
        const florence2NumBeamsWidget = node.widgets?.find(w => w.name === "florence2_num_beams");
        const yoloIouWidget = node.widgets?.find(w => w.name === "yolo_iou");
        const yoloAgnosticNmsWidget = node.widgets?.find(w => w.name === "yolo_agnostic_nms");
        const yoloMaxDetWidget = node.widgets?.find(w => w.name === "yolo_max_det");

        log("Found detector parameter widgets:", {
            text_threshold: !!textThresholdWidget,
            florence2_max_tokens: !!florence2MaxTokensWidget,
            florence2_num_beams: !!florence2NumBeamsWidget,
            yolo_iou: !!yoloIouWidget,
            yolo_agnostic_nms: !!yoloAgnosticNmsWidget,
            yolo_max_det: !!yoloMaxDetWidget
        });

        // Function to update widget visibility based on connected model
        const updateDetectorWidgets = (modelType) => {
            log("Updating detector widgets for model type:", modelType);

            // GroundingDINO parameters (text_threshold)
            if (textThresholdWidget) {
                if (modelType === "grounding_dino") {
                    showWidget(node, textThresholdWidget);
                } else {
                    hideWidget(node, textThresholdWidget);
                }
            }

            // Florence-2 parameters (inference-time only)
            if (florence2MaxTokensWidget) {
                if (modelType === "florence2") {
                    showWidget(node, florence2MaxTokensWidget);
                } else {
                    hideWidget(node, florence2MaxTokensWidget);
                }
            }

            if (florence2NumBeamsWidget) {
                if (modelType === "florence2") {
                    showWidget(node, florence2NumBeamsWidget);
                } else {
                    hideWidget(node, florence2NumBeamsWidget);
                }
            }

            // YOLO-World parameters
            if (yoloIouWidget) {
                if (modelType === "yolo_world") {
                    showWidget(node, yoloIouWidget);
                } else {
                    hideWidget(node, yoloIouWidget);
                }
            }

            if (yoloAgnosticNmsWidget) {
                if (modelType === "yolo_world") {
                    showWidget(node, yoloAgnosticNmsWidget);
                } else {
                    hideWidget(node, yoloAgnosticNmsWidget);
                }
            }

            if (yoloMaxDetWidget) {
                if (modelType === "yolo_world") {
                    showWidget(node, yoloMaxDetWidget);
                } else {
                    hideWidget(node, yoloMaxDetWidget);
                }
            }

            // Force canvas redraw
            node.setDirtyCanvas(true, true);
            if (app.graph) {
                app.graph.setDirtyCanvas(true, true);
            }

            requestAnimationFrame(() => {
                const newSize = node.computeSize();
                node.setSize([node.size[0], newSize[1]]);
                node.setDirtyCanvas(true, true);
                if (app.canvas) {
                    app.canvas.setDirty(true, true);
                }

                requestAnimationFrame(() => {
                    if (app.canvas) {
                        app.canvas.draw(true, true);
                    }
                    log("Detector widget visibility update complete");
                });
            });
        };

        // Function to get model type from connected loader
        const getConnectedModelType = () => {
            // Find the model input (slot 0)
            const modelInput = node.inputs?.find(input => input.name === "model");
            if (!modelInput || !modelInput.link) {
                log("No model connected to detector");
                return null;
            }

            // Get the connected link
            const link = app.graph.links[modelInput.link];
            if (!link) {
                log("Link not found");
                return null;
            }

            // Get the connected node (the loader)
            const loaderNode = app.graph.getNodeById(link.origin_id);
            if (!loaderNode || loaderNode.comfyClass !== "GroundingModelLoader") {
                log("Connected node is not a GroundingModelLoader");
                return null;
            }

            // Get the model widget value from the loader
            const loaderModelWidget = loaderNode.widgets?.find(w => w.name === "model");
            if (!loaderModelWidget) {
                log("Model widget not found in loader");
                return null;
            }

            const selectedModel = loaderModelWidget.value;
            log("Connected loader has model:", selectedModel);

            return getModelType(selectedModel);
        };

        // Update widgets when a connection is made
        const origOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function(type, index, connected, link_info) {
            log("Detector connection changed:", type, index, connected);

            if (origOnConnectionsChange) {
                origOnConnectionsChange.apply(this, arguments);
            }

            // If model input was connected/disconnected
            if (type === 1 && index === 0) {  // type 1 = input, index 0 = model input
                if (connected) {
                    const modelType = getConnectedModelType();
                    if (modelType) {
                        updateDetectorWidgets(modelType);
                    }
                } else {
                    // Disconnected - show all widgets? Or hide all? Let's hide all for now
                    log("Model disconnected, hiding all model-specific widgets");
                    updateDetectorWidgets("unknown");
                }
            }
        };

        // Check if there's already a connection and update accordingly
        setTimeout(() => {
            const modelType = getConnectedModelType();
            if (modelType) {
                log("Initial detector setup with connected model type:", modelType);
                updateDetectorWidgets(modelType);
            } else {
                log("No initial connection, hiding all model-specific widgets");
                updateDetectorWidgets("unknown");
            }
        }, 150);

        // Also listen for changes in the connected loader's model selection
        // We need to hook into graph events for this
        const checkLoaderChanges = setInterval(() => {
            if (!node.graph) {
                clearInterval(checkLoaderChanges);
                return;
            }

            const modelType = getConnectedModelType();
            if (modelType && modelType !== node._lastModelType) {
                log("Detected model type change in connected loader:", modelType);
                node._lastModelType = modelType;
                updateDetectorWidgets(modelType);
            }
        }, 500);

        // Clean up interval when node is removed
        const origOnRemoved = node.onRemoved;
        node.onRemoved = function() {
            clearInterval(checkLoaderChanges);
            if (origOnRemoved) {
                origOnRemoved.apply(this, arguments);
            }
        };
    },

    setupMaskModelLoader(node) {
        log("Setting up GroundingMaskModelLoader node");

        // Find the model widget
        const modelWidget = node.widgets?.find(w => w.name === "model");
        if (!modelWidget) {
            log("ERROR: Model widget not found!");
            log("Available widgets:", node.widgets?.map(w => w.name));
            return;
        }

        log("Mask model widget found, current value:", modelWidget.value);

        // Find attention parameter widget
        const florence2AttnWidget = node.widgets?.find(w => w.name === "florence2_attn");

        log("Found parameter widgets:", {
            florence2_attn: !!florence2AttnWidget
        });

        // Function to update widget visibility based on selected model
        const updateWidgetVisibility = (selectedModel) => {
            const modelType = getModelType(selectedModel);
            log("Updating mask model widget visibility for model:", selectedModel, "type:", modelType);

            // Florence-2 parameters (load-time)
            if (florence2AttnWidget) {
                if (modelType === "florence2") {
                    showWidget(node, florence2AttnWidget);
                } else {
                    hideWidget(node, florence2AttnWidget);
                }
            }

            // Force canvas redraw
            node.setDirtyCanvas(true, true);
            if (app.graph) {
                app.graph.setDirtyCanvas(true, true);
            }

            requestAnimationFrame(() => {
                const newSize = node.computeSize();
                node.setSize([node.size[0], newSize[1]]);
                node.setDirtyCanvas(true, true);
                if (app.canvas) {
                    app.canvas.setDirty(true, true);
                }

                requestAnimationFrame(() => {
                    if (app.canvas) {
                        app.canvas.draw(true, true);
                    }
                    log("Mask model widget visibility update complete");
                });
            });
        };

        // Store original callback
        const origCallback = modelWidget.callback;

        // Override callback to update visibility when model changes
        modelWidget.callback = function(value) {
            log("Mask model changed to:", value);

            // Call original callback if it exists
            const result = origCallback?.apply(this, arguments);

            // Update widget visibility
            updateWidgetVisibility(value);

            return result;
        };

        // Initialize visibility on node creation
        log("Initializing mask model widget visibility...");
        updateWidgetVisibility(modelWidget.value);
    },

    setupMaskDetector(node) {
        log("Setting up GroundingMaskDetector node");

        // Find all model-specific parameter widgets
        const florence2MaxTokensWidget = node.widgets?.find(w => w.name === "florence2_max_tokens");
        const florence2NumBeamsWidget = node.widgets?.find(w => w.name === "florence2_num_beams");
        const sa2vaMaxTokensWidget = node.widgets?.find(w => w.name === "sa2va_max_tokens");
        const sa2vaNumBeamsWidget = node.widgets?.find(w => w.name === "sa2va_num_beams");

        log("Found mask detector parameter widgets:", {
            florence2_max_tokens: !!florence2MaxTokensWidget,
            florence2_num_beams: !!florence2NumBeamsWidget,
            sa2va_max_tokens: !!sa2vaMaxTokensWidget,
            sa2va_num_beams: !!sa2vaNumBeamsWidget
        });

        // Function to update widget visibility based on connected model
        const updateMaskDetectorWidgets = (modelType) => {
            log("Updating mask detector widgets for model type:", modelType);

            // Florence-2 Seg parameters
            if (florence2MaxTokensWidget) {
                if (modelType === "florence2") {
                    showWidget(node, florence2MaxTokensWidget);
                } else {
                    hideWidget(node, florence2MaxTokensWidget);
                }
            }

            if (florence2NumBeamsWidget) {
                if (modelType === "florence2") {
                    showWidget(node, florence2NumBeamsWidget);
                } else {
                    hideWidget(node, florence2NumBeamsWidget);
                }
            }

            // SA2VA parameters
            if (sa2vaMaxTokensWidget) {
                if (modelType === "sa2va") {
                    showWidget(node, sa2vaMaxTokensWidget);
                } else {
                    hideWidget(node, sa2vaMaxTokensWidget);
                }
            }

            if (sa2vaNumBeamsWidget) {
                if (modelType === "sa2va") {
                    showWidget(node, sa2vaNumBeamsWidget);
                } else {
                    hideWidget(node, sa2vaNumBeamsWidget);
                }
            }

            // Force canvas redraw
            node.setDirtyCanvas(true, true);
            if (app.graph) {
                app.graph.setDirtyCanvas(true, true);
            }

            requestAnimationFrame(() => {
                const newSize = node.computeSize();
                node.setSize([node.size[0], newSize[1]]);
                node.setDirtyCanvas(true, true);
                if (app.canvas) {
                    app.canvas.setDirty(true, true);
                }

                requestAnimationFrame(() => {
                    if (app.canvas) {
                        app.canvas.draw(true, true);
                    }
                    log("Mask detector widget visibility update complete");
                });
            });
        };

        // Helper function to detect SA2VA model type
        const getMaskModelType = (modelName) => {
            if (modelName.startsWith("Florence-2:")) {
                return "florence2";
            }
            if (modelName.startsWith("SA2VA:")) {
                return "sa2va";
            }
            return "unknown";
        };

        // Function to get model type from connected mask loader
        const getConnectedMaskModelType = () => {
            // Find the model input (slot 0)
            const modelInput = node.inputs?.find(input => input.name === "model");
            if (!modelInput || !modelInput.link) {
                log("No model connected to mask detector");
                return null;
            }

            // Get the connected link
            const link = app.graph.links[modelInput.link];
            if (!link) {
                log("Link not found");
                return null;
            }

            // Get the connected node (the loader)
            const loaderNode = app.graph.getNodeById(link.origin_id);
            if (!loaderNode || loaderNode.comfyClass !== "GroundingMaskModelLoader") {
                log("Connected node is not a GroundingMaskModelLoader");
                return null;
            }

            // Get the model widget value from the loader
            const loaderModelWidget = loaderNode.widgets?.find(w => w.name === "model");
            if (!loaderModelWidget) {
                log("Model widget not found in loader");
                return null;
            }

            const selectedModel = loaderModelWidget.value;
            log("Connected mask loader has model:", selectedModel);

            return getMaskModelType(selectedModel);
        };

        // Update widgets when a connection is made
        const origOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function(type, index, connected, link_info) {
            log("Mask detector connection changed:", type, index, connected);

            if (origOnConnectionsChange) {
                origOnConnectionsChange.apply(this, arguments);
            }

            // If model input was connected/disconnected
            if (type === 1 && index === 0) {  // type 1 = input, index 0 = model input
                if (connected) {
                    const modelType = getConnectedMaskModelType();
                    if (modelType) {
                        updateMaskDetectorWidgets(modelType);
                    }
                } else {
                    // Disconnected - hide all model-specific widgets
                    log("Model disconnected, hiding all model-specific widgets");
                    updateMaskDetectorWidgets("unknown");
                }
            }
        };

        // Check if there's already a connection and update accordingly
        setTimeout(() => {
            const modelType = getConnectedMaskModelType();
            if (modelType) {
                log("Initial mask detector setup with connected model type:", modelType);
                updateMaskDetectorWidgets(modelType);
            } else {
                log("No initial connection, hiding all model-specific widgets");
                updateMaskDetectorWidgets("unknown");
            }
        }, 150);

        // Also listen for changes in the connected loader's model selection
        const checkLoaderChanges = setInterval(() => {
            if (!node.graph) {
                clearInterval(checkLoaderChanges);
                return;
            }

            const modelType = getConnectedMaskModelType();
            if (modelType && modelType !== node._lastModelType) {
                log("Detected model type change in connected mask loader:", modelType);
                node._lastModelType = modelType;
                updateMaskDetectorWidgets(modelType);
            }
        }, 500);

        // Clean up interval when node is removed
        const origOnRemoved = node.onRemoved;
        node.onRemoved = function() {
            clearInterval(checkLoaderChanges);
            if (origOnRemoved) {
                origOnRemoved.apply(this, arguments);
            }
        };

        log("GroundingMaskDetector setup complete");
    }
});

log("ComfyUI-Grounding dynamic parameters extension loaded");
