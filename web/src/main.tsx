import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";

type ErrorBoundaryState = {
  hasError: boolean;
  message: string;
};

class RootErrorBoundary extends React.Component<
  React.PropsWithChildren,
  ErrorBoundaryState
> {
  constructor(props: React.PropsWithChildren) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      message: error?.message ?? "Unknown runtime error",
    };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Keep stack traces in the dev console while rendering a safe UI fallback.
    console.error("Root render failure:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: "100vh",
            margin: 0,
            padding: "24px",
            fontFamily: "\"Roboto Condensed\", system-ui, -apple-system, sans-serif",
            background: "radial-gradient(circle at 20% 20%, #121826, #0a0d14 60%)",
            color: "#fbb6b8",
          }}
        >
          <h1 style={{ marginTop: 0 }}>UI Runtime Error</h1>
          <p style={{ color: "#dfe6f5" }}>
            The app crashed before render. Check browser console for stack trace.
          </p>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              background: "rgba(59, 15, 21, 0.7)",
              border: "1px solid #7f1d1d",
              borderRadius: "8px",
              padding: "12px",
            }}
          >
            {this.state.message}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </React.StrictMode>,
);
