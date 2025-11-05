import React, { useState, useEffect } from 'react'
import styles from './PlaygroundPage.module.css'

const PlaygroundPage: React.FC = () => {
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  // Use the backend proxy route
  const openWebUIUrl = '/embedded/openwebui/'

  useEffect(() => {
    // Check if Open WebUI is available
    fetch('/embedded/openwebui/', { method: 'HEAD' })
      .then((response) => {
        if (response.ok || response.status === 404) {
          // 404 is OK, means proxy is configured but endpoint might not be
          setError(null)
        } else if (response.status === 503) {
          setError('Open WebUI is not configured. Please set TARGET_OPENWEBUI_URL environment variable.')
        } else {
          setError(`Unable to connect to Open WebUI: ${response.statusText}`)
        }
      })
      .catch((err) => {
        setError('Unable to connect to Open WebUI. Please ensure it is running and configured.')
        console.error('Open WebUI connection error:', err)
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])

  return (
    <div className={styles.container}>
      <div className={styles.iframeContainer}>
        {loading && (
          <div className={styles.placeholder}>
            <span className={styles.placeholderIcon}>⏳</span>
            <h3>Loading Playground...</h3>
          </div>
        )}

        {error && !loading && (
          <div className={styles.placeholder}>
            <span className={styles.placeholderIcon}>⚠️</span>
            <h3>Open WebUI Playground</h3>
            <p className={styles.error}>{error}</p>
            <p className={styles.note}>
              To enable the playground:
              <br />
              1. Start Open WebUI on a port (e.g., 3001)
              <br />
              2. Set TARGET_OPENWEBUI_URL environment variable:
              <br />
              <code>export TARGET_OPENWEBUI_URL=http://localhost:3001</code>
              <br />
              3. Restart the dashboard backend
            </p>
          </div>
        )}

        {!error && !loading && (
          <iframe
            src={openWebUIUrl}
            className={styles.iframe}
            title="Open WebUI Playground"
            allowFullScreen
            onError={() => {
              setError('Failed to load Open WebUI. Please check the configuration.')
              setLoading(false)
            }}
          />
        )}
      </div>
    </div>
  )
}

export default PlaygroundPage
