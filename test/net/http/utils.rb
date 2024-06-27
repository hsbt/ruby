require 'socket'
require 'openssl'

module TestNetHTTPUtils
  def start(&block)
    @http ||= new_http
    yield @http if block_given?
  end

  def new_http
    http = Net::HTTP.new(config('host'), config('port'), config('proxy_host'), config('proxy_port'))
    http.use_ssl = true if config('ssl_enable')
    http.set_debug_output(logfile)
    http
  end

  def config(key)
    @config ||= self.class::CONFIG
    @config[key]
  end

  def logfile
    $DEBUG ? $stderr : NullWriter.new
  end

  def setup
    spawn_server
  end

  def teardown
    if @server
      @server.close
      @server_thread.join
    end
    @log_tester.call(@log) if @log_tester
  end

  def spawn_server
    @log = []
    @log_tester = lambda { |log| assert_equal([], log) }
    @config = self.class::CONFIG

    server = TCPServer.new(config('host'), 0)  # 0 for any available port
    if config('ssl_enable')
      ssl_context = OpenSSL::SSL::SSLContext.new
      ssl_context.cert = OpenSSL::X509::Certificate.new(File.open(config('ssl_certificate')).read)
      ssl_context.key = OpenSSL::PKey::RSA.new(File.open(config('ssl_private_key')).read)
      server = OpenSSL::SSL::SSLServer.new(server, ssl_context)
    end

    @server_thread = Thread.new do
      loop do
        client = server.accept
        handle_request(client)
      end
    end

    @config['port'] = server.addr[1]
  end

  def handle_request(client)
    request_line = client.gets
    method, path, _ = request_line.split

    case method
    when 'GET'
      client.print "HTTP/1.1 200 OK\r\n"
      client.print "Content-Type: #{config('chunked') ? 'application/octet-stream' : 'text/plain'}\r\n"
      client.print "\r\n"
      client.print $test_net_http_data
    when 'POST'
      content_length = client.gets.split(': ')[1].to_i
      client.read(content_length)  # consume the body
      client.print "HTTP/1.1 200 OK\r\n"
      client.print "Content-Type: text/plain\r\n"
      client.print "X-request-uri: /\r\n"
      client.print "\r\n"
      client.print "Received POST data."
    when 'PATCH'
      content_length = client.gets.split(': ')[1].to_i
      client.read(content_length)  # consume the body
      client.print "HTTP/1.1 200 OK\r\n"
      client.print "Content-Type: text/plain\r\n"
      client.print "\r\n"
      client.print "Received PATCH data."
    else
      client.puts "HTTP/1.1 405 Method Not Allowed\r\n"
    end

    client.close
  end

  class NullWriter
    def <<(s) end
    def puts(*args) end
    def print(*args) end
    def printf(*args) end
  end

  def self.clean_http_proxy_env
    orig = {
      'http_proxy'      => ENV['http_proxy'],
      'http_proxy_user' => ENV['http_proxy_user'],
      'http_proxy_pass' => ENV['http_proxy_pass'],
      'no_proxy'        => ENV['no_proxy'],
    }

    orig.each_key do |key|
      ENV.delete key
    end

    yield
  ensure
    orig.each do |key, value|
      ENV[key] = value
    end
  end
end
