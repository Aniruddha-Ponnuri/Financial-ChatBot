import { render, screen } from '@testing-library/react';
import App from './App';

jest.mock('./Chatbot', () => () => <div data-testid="chatbot">Mock Chatbot</div>);

test('renders app shell with chatbot', () => {
  render(<App />);
  expect(screen.getAllByTestId('chatbot').length).toBeGreaterThan(0);
});
