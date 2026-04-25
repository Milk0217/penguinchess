// Vitest setup file
import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom'

// Clean up after each test
afterEach(() => {
  cleanup()
})